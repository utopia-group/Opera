from typing import Any, Sequence

from frozendict import frozendict

import b2s.ast_extra as ast
from b2s.const import VAR_NAME
from b2s.lang import (
    BinOpKinds,
    EBinOp,
    ECall,
    EIte,
    ELam,
    ELet,
    EMapGet,
    EMapNil,
    EMapUpdate,
    ENil,
    EPair,
    EPythonExpr,
    EStream,
    EUnknown,
    EVar,
    Expr,
    Id,
    expr_is_complete,
    fold_expr,
    replace_unknown,
    to_const_expr,
)
from b2s.utils import parse_term


def _replace_var_with_state_map(py_expr: str, env_vars: set[str]) -> str:
    class StateMapReplaceTransformer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> Any:
            if node.id in env_vars:
                return node
            key_name = "'" + ast.unparse(node) + "'"
            return parse_term(f"{VAR_NAME.STATE_MAP}[{key_name}]")

    t = parse_term(py_expr)
    t = StateMapReplaceTransformer().visit(t)
    return ast.unparse(t)


class ExtractInitializers(ast.NodeVisitor):
    initializers: dict[str, Any]

    def __init__(self) -> None:
        self.initializers = {}
        super().__init__()

    def visit_Assign(self, node: ast.Assign) -> Any:
        match (node.targets, node.value):
            case ([ast.Name(id=var_name)], ast.Constant(value=init_val)):
                self.initializers[var_name] = init_val
            case (_, ast.Constant(value=_)):
                raise NotImplementedError(f"TBD: {ast.unparse(node)}")
            case ([ast.Tuple(targets)], ast.Tuple(consts)):
                assert len(targets) == len(consts), "length mismatch"
                for target, elt in zip(targets, consts):
                    assert isinstance(target, ast.Name)
                    assert isinstance(elt, ast.Constant)
                    self.initializers[target.id] = elt.value

    @staticmethod
    def extract(stmt: Any) -> dict[str, Any]:
        visitor = ExtractInitializers()
        visitor.visit(stmt)
        return visitor.initializers


class PyAstToIntermLang(ast.NodeVisitor):
    binop_obj_to_kind: dict[type, BinOpKinds] = {
        ast.Add: BinOpKinds.ADD,
        ast.Sub: BinOpKinds.SUB,
        ast.Mult: BinOpKinds.MUL,
        ast.Div: BinOpKinds.DIV,
        ast.Eq: BinOpKinds.EQ,
        ast.NotEq: BinOpKinds.NEQ,
        ast.Lt: BinOpKinds.LT,
        ast.Gt: BinOpKinds.GT,
        ast.LtE: BinOpKinds.LTE,
        ast.GtE: BinOpKinds.GTE,
        ast.And: BinOpKinds.AND,
        ast.Or: BinOpKinds.OR,
        ast.Pow: BinOpKinds.POW,
        ast.FloorDiv: BinOpKinds.DIV,
    }

    @staticmethod
    def convert(py_src: str, func: str, stream_param: str) -> Expr:
        src_ast = ast.parse(py_src)
        return PyAstToIntermLang(func, stream_param).visit(src_ast)

    func: str
    stream_param: str
    env_vars: set[str]
    reserved_vars: set[str]

    def __init__(self, func: str, stream_param: str) -> None:
        self.func = func
        self.stream_param = stream_param
        self.env_vars = {
            stream_param,
            func,
            "len",
            "list",
            "dict",
            "log",
            "exp",
            "abs",
            "int",
            "float",
        }
        self.reserved_vars = {"list", "dict", "defaultdict", "create_dict"}

    def visit_Constant(self, node: ast.Constant) -> Any:
        return to_const_expr(node.value)

    def visit_Module(self, node: ast.Module) -> Expr:
        matched = list(
            filter(
                lambda x: isinstance(x, ast.FunctionDef) and x.name == self.func,
                node.body,
            )
        )

        assert len(matched) == 1, "one and only one function should be found"
        assert isinstance(matched[0], ast.FunctionDef)
        return self.visit_FunctionDef(matched[0])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Expr:
        param_names = tuple(param.arg for param in node.args.args)
        self.env_vars.update(param_names)
        body = ELet(VAR_NAME.STATE_MAP, EMapNil(), self.convert_stmt_seq(node.body))

        assert (
            self.stream_param in param_names
        ), "stream parameter should be in the function parameters"
        return ELam(param_names, body)

    def visit_For_alt(self, node: ast.For, foldl_state: Expr) -> Expr:
        iter_expr: Expr
        match node.iter:
            case ast.Name(id=self.stream_param):
                iter_expr = EStream(self.stream_param)
            case ast.Name(id=other):
                iter_expr = EVar(other)
            case other:
                iter_expr = EPythonExpr(
                    _replace_var_with_state_map(ast.unparse(other), self.env_vars)
                )

        match node.target:
            case ast.Name(id=loop_var):
                pass
            case _:
                raise NotImplementedError("Unsupported loop target")

        loop_update_func = self.convert_stmt_seq(node.body, False)
        loop_update_func = replace_unknown(
            loop_update_func, "next", EVar(VAR_NAME.STATE_MAP)
        )

        return ECall(
            EVar("foldl"),
            (
                ELam((VAR_NAME.STATE_MAP, loop_var), loop_update_func),
                foldl_state,
                iter_expr,
            ),
        )

    def visit_Assign(self, node: ast.Assign) -> Any:
        match node:
            case ast.Assign(targets=[ast.Name(id=var_name)], value=value):
                return (var_name, self.visit(value))

        raise NotImplementedError(f"TBD: {ast.unparse(node)}")

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        match node:
            case ast.AugAssign(target=ast.Name(n), value=value, op=op):
                return (
                    n,
                    EBinOp(
                        EVar(n),
                        self.binop_obj_to_kind[type(op)],
                        self.visit(value),
                    ),
                )
        raise NotImplementedError(f"TBD: {ast.unparse(node)}")

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id == self.stream_param:
            return EStream(node.id)
        return EVar(node.id)

    def visit_Compare(self, node: ast.Compare) -> Any:
        assert (
            len(node.comparators) == len(node.ops) == 1
        ), "only one comparator is supported"
        if type(node.ops[0]) not in self.binop_obj_to_kind:
            return self.generic_visit(node)
        return EBinOp(
            self.visit(node.left),
            self.binop_obj_to_kind[type(node.ops[0])],
            self.visit(node.comparators[0]),
        )

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return EPair(tuple(self.visit(elt) for elt in node.elts))

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        return EBinOp(
            self.visit(node.left),
            self.binop_obj_to_kind[type(node.op)],
            self.visit(node.right),
        )

    def visit_Call(self, node: ast.Call) -> Any:
        match node.func:
            case ast.Name(id=n) if n in self.reserved_vars:
                return EPythonExpr(ast.unparse(node))
            case ast.Name():
                return ECall(
                    self.visit(node.func), tuple(self.visit(arg) for arg in node.args)
                )
            case ast.Attribute():
                return EPythonExpr(ast.unparse(node))

    def visit_If(self, node: ast.If) -> dict[str, EIte]:
        assert len(node.orelse) <= 1, "else branch should be at most one stmt"
        body = self.convert_stmt_seq(node.body, False)
        assert isinstance(body, EMapUpdate)
        body_updates = body.updates

        orelse = None
        orelse_updates: frozendict[Id, Expr] = frozendict()
        if node.orelse:
            orelse = self.convert_stmt_seq(node.orelse, False)
            assert isinstance(orelse, EMapUpdate)
            orelse_updates = orelse.updates

        cond_var_updates: dict[str, EIte] = {}
        for v in body_updates.keys() | orelse_updates.keys():
            cond_var_updates[v] = EIte(
                self.visit(node.test),
                body_updates.get(v, EVar(v)),
                orelse_updates.get(v, EVar(v)),
            )
        return cond_var_updates

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        match node.op, node.operand:
            case ast.USub(), ast.Constant(value=v):
                return to_const_expr(-v)
            case ast.USub(), _:
                return EBinOp(
                    to_const_expr(-1), BinOpKinds.MUL, self.visit(node.operand)
                )

        raise NotImplementedError(f"TBD: {ast.unparse(node)}")

    def visit_Dict(self, node: ast.Dict) -> Any:
        return EPythonExpr(ast.unparse(node))

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        match node.slice:
            case ast.Constant(value=v):
                match v:
                    case 0:
                        return ECall(EVar("fst"), (self.visit(node.value),))
                    case 1:
                        return ECall(EVar("snd"), (self.visit(node.value),))

        return self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> Any:
        return EPythonExpr(ast.unparse(node))

    def _update_stmts_to_rules(
        self, m: Expr, update_stmts: list[tuple[str, Expr]]
    ) -> EMapUpdate:
        rules: dict[str, Expr] = {}
        for v, expr in update_stmts:
            if v not in rules:
                rules[v] = expr
                continue

            rules[v] = fold_expr(
                expr,
                f_var=lambda _: rules[v],
                f_stream=lambda x: x,
                f_int=lambda x: x,
                f_bool=lambda x: x,
                f_str=lambda x: x,
                f_binop=lambda left, op, right: EBinOp(left, op, right),
                f_call=lambda func, args: ECall(func, args),
                f_lam=lambda params, body: ELam(params, body),
                f_let=lambda name, expr, body: ELet(name, expr, body),
                f_ite=lambda cond, then, els: EIte(cond, then, els),
                f_nil=lambda: ENil(),
                f_pair=lambda elts: EPair(elts),
                f_unk=lambda unk_id_: EUnknown(unk_id_),
                f_map=lambda e, updates: EMapUpdate(e, updates),
                f_map_nil=lambda: EMapNil(),
                f_map_get=lambda e, k: EMapGet(e, k),
                f_python_expr=lambda e: EPythonExpr(e),
            )

        return EMapUpdate.create(m, **rules)

    def convert_stmt_seq(
        self,
        stmts: Sequence[Any],
        is_complete: bool = True,
    ) -> Expr:
        next_id = "next"
        next_node = EUnknown(next_id)
        prog: Expr = next_node

        current_update_stmts: list[tuple[str, Expr]] = []

        for stmt in stmts:
            match stmt:
                case ast.Assign() | ast.AugAssign():
                    current_update_stmts.append(self.visit(stmt))

                case ast.If():
                    cond_var_updates: dict[str, EIte] = self.visit_If(stmt)
                    current_update_stmts.extend(cond_var_updates.items())

                case ast.For():
                    init_state = self._update_stmts_to_rules(
                        EVar(VAR_NAME.STATE_MAP), current_update_stmts
                    )
                    loop_expr = self.visit_For_alt(stmt, init_state)
                    prog = replace_unknown(
                        prog, next_id, ELet(VAR_NAME.STATE_MAP, loop_expr, next_node)
                    )
                    current_update_stmts.clear()

                case ast.Return(value=value):
                    assert value is not None, "return value should not be None"
                    ret_val = self.visit(value)
                    ret_val = fold_expr(
                        ret_val,
                        f_var=lambda v: EMapGet(EVar(VAR_NAME.STATE_MAP), v.name)
                        if v.name not in self.env_vars
                        else v,
                        f_stream=lambda x: x,
                        f_int=lambda x: x,
                        f_bool=lambda x: x,
                        f_str=lambda x: x,
                        f_binop=lambda left, op, right: EBinOp(left, op, right),
                        f_call=lambda func, args: ECall(func, args),
                        f_lam=lambda params, body: ELam(params, body),
                        f_let=lambda name, expr, body: ELet(name, expr, body),
                        f_ite=lambda cond, then, els: EIte(cond, then, els),
                        f_nil=lambda: ENil(),
                        f_pair=lambda elts: EPair(elts),
                        f_unk=lambda unk_id_: EUnknown(unk_id_),
                        f_map=lambda e, updates: EMapUpdate(e, updates),  # type: ignore
                        f_map_nil=lambda: EMapNil(),
                        f_map_get=lambda e, k: EMapGet(e, k),
                        f_python_expr=lambda e: EPythonExpr(e),
                    )
                    assert isinstance(
                        ret_val, Expr
                    ), "return value should be an expression"
                    if current_update_stmts:
                        prog = replace_unknown(
                            prog,
                            next_id,
                            ELet(
                                VAR_NAME.STATE_MAP,
                                self._update_stmts_to_rules(
                                    EVar(VAR_NAME.STATE_MAP), current_update_stmts
                                ),
                                next_node,
                            ),
                        )
                        current_update_stmts.clear()
                    prog = replace_unknown(prog, next_id, ret_val)

                case ast.Expr(value=ast.Constant(value=_)):
                    pass  # comment/docstring

                case _:
                    raise NotImplementedError(f"TBD: {ast.unparse(stmt)}")

        if current_update_stmts:
            prog = replace_unknown(
                prog,
                next_id,
                self._update_stmts_to_rules(
                    EVar(VAR_NAME.STATE_MAP), current_update_stmts
                ),
            )
        assert (
            expr_is_complete(prog) or not is_complete
        ), "expression should be complete"
        return prog
