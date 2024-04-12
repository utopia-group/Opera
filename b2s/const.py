class VAR_NAME:  # pylint: disable=too-few-public-methods
    PREVIOUS_PREFIX = "_prev_"
    CURRENT_PREFIX = "_cur_"
    CURRENT_OUTPUT = "_cur_out"
    PREVIOUS_OUTPUT = "prev_out"
    PREVIOUS_LENGTH = "prev_len"
    CURRENT_PROG_OUT = "cur_prog_out"

    CURRENT_ELEMENT = "x__"
    CURRENT_ELEMENT_1 = "x__1"
    CURRENT_ELEMENT_2 = "x__2"
    PREVIOUS_LIST = "_init"
    INPUT_STREAM = "xs"
    PRE_STATES = "_pre_states"
    POST_PREUNROLL_STATES = "_post_states_preunroll"
    POST_POSTUNROLL_STATES = "_post_states_postunroll"

    STATE_BEFORE_LOOP = "_before_loop"
    STATE_AFTER_LOOP_PREUNROLL = "_after_loop_preunroll"
    STATE_AFTER_LOOP_POSTUNROLL = "_after_loop_postunroll"

    STATE_UNKNOWNS = "_unknowns"
    RET_TEMP_VAR = "__ret_val"

    STATE_MAP = "S"

    RESERVERD_NAMES = {
        CURRENT_OUTPUT,
        PREVIOUS_OUTPUT,
        CURRENT_ELEMENT,
        PREVIOUS_LIST,
        PRE_STATES,
        POST_PREUNROLL_STATES,
        POST_POSTUNROLL_STATES,
        STATE_BEFORE_LOOP,
        STATE_AFTER_LOOP_PREUNROLL,
        STATE_AFTER_LOOP_POSTUNROLL,
    }


class MAGIC_ID:  # pylint: disable=too-few-public-methods
    RETURN = "@@return@@"
    SPLITTER = "@@<<TEXAS>>@@"


SYN_OUTPUT_VAR_KEY = "expected_output"
