import os
from typing import Optional

import openai

import b2s.ast_extra as ast
from b2s import utils
from b2s.input_config import ImperativeRelationalSignature, InputConfig

openai.api_key = os.getenv("OPENAI_KEY")

SYSTEM_PROMPT = r"""
You are a helpful AI programming assistant. Follow the
user's requirements carefully.
"""

PROMPT_TEMPLATE = r"""
Here is a implementation of an offline algorithm that takes
an input stream `xs` as its input:

```python
def mean(xs):
    s = 0
    for n in xs:
        s += n
    return s / len(xs)
```

Without using the input stream `xs` in the function signature,
the equivalent online version is:

```python
def mean_online(prev_avg, prev_len, x):
    l = prev_len + 1
    s = (prev_avg*prev_len + x) / l
    return s, l
```

Here is a implementation of an offline algorithm that takes
an input stream `xs` as its input:

```python
def nobs(xs):
    n = 0
    for _ in xs:
        n += 1
    return n
```

Without using the input stream `xs` in the function signature,
the equivalent online version is:

```python
def nobs_online(prev_out, x):
    return prev_out + 1
```


Here is a implementation of an offline algorithm that takes
an input stream `{stream_param}` as its input:

```python
{offline}
```

Without using the input stream `{stream_param}` in the function signature,
the equivalent online version is:
"""

REL_SIG_ONE_SHOT_PROMPT = r"""
Background:
(Relational Signature) Given offline P and online P', a relational signature
is a formula of the form \bigwedge_i y_i = f_i(xs)

Definition (Equivalence modulo relational signature):
We say P and P' are equivalent modulo Phi, denoted P \equiv_\Phi P'. if the
following Hoare triple is valid:

{P(xs) = fst(Y) & Phi(xs, Y) & xs' = xs ++ [x]}
    Y' := P'(x, Y)
{P(xs') = fst(Y') & Phi(xs', Y')}

Inductive relational signature: We say that a relational signature Phi for P, P' is inductive if
(1) \exists I. Phi([], I) /\ P([]) = fst(I)
(2) P \equiv_Phi P'

Question:
Here is a implementation of an offline algorithm that takes
an input stream xs as the input:

{offline}

Given the variable names below, derive a relational signature
that is inductive for the offline algorithm above.

{params}"""


class GptConverter:
    gen_sig: bool
    model: str
    max_tokens: int

    def __init__(
        self, gpt_gen_sig: bool, gpt_model: str, gpt_max_tokens: int, **_
    ) -> None:
        self.gen_sig = gpt_gen_sig
        self.model = gpt_model
        self.max_tokens = gpt_max_tokens

    @staticmethod
    def find_function_by_name(
        module: ast.Module, name: str
    ) -> Optional[ast.FunctionDef]:
        """
        Find a function by its name.
        """
        for node in ast.walk(module):
            match node:
                case ast.FunctionDef() if node.name == name:
                    return node
        return None

    def generate_rel_sig(
        self, func_str: str, spec: ImperativeRelationalSignature
    ) -> str:
        params = list(spec.keys())
        params_str = ", ".join(params)

        prompt = REL_SIG_ONE_SHOT_PROMPT.format(
            offline=func_str,
            params=params_str,
        )

        comp = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=1,
            max_tokens=self.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        return comp.choices[0]["message"]["content"]  # type: ignore

    def generate_solution(
        self, func_str: str, stream_param: str, spec: ImperativeRelationalSignature
    ) -> str:
        params = list(spec.keys())
        params_str = ", ".join(params)
        new_params_str = ", ".join([f"new_{n}" for n in params])

        prompt = PROMPT_TEMPLATE.format(
            offline=func_str,
            params=params_str,
            new_params=new_params_str,
            stream_param=stream_param,
        )

        comp = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful programming assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=1,
            max_tokens=self.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        return comp.choices[0]["message"]["content"]  # type: ignore

    def convert(self, src: str) -> Optional[str]:
        module = ast.parse(src)
        docstr = utils.get_docstring(module)
        assert docstr is not None, "Cannot find docstring"
        input_truth = InputConfig.load_from_docstring(docstr)

        func = self.find_function_by_name(module, input_truth.func)
        assert func is not None, f"Cannot find function {input_truth.func}"

        func.name = "offline"
        offline_str = ast.unparse(func)
        if self.gen_sig:
            return self.generate_rel_sig(offline_str, input_truth.relational_sig)
        else:
            return self.generate_solution(
                offline_str, input_truth.stream_param, input_truth.relational_sig
            )
