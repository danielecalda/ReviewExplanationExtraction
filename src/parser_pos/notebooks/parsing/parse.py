from parsing.rule import Rule, is_cat, is_lexical
from collections import Iterable
from types import FunctionType
from six import StringIO


class Parse:
    def __init__(self, rule, children):
        self.rule = rule
        self.children = tuple(children[:])
        self.semantics = compute_semantics(self)
        self.score = float('NaN')
        self.denotation = None
        validate_parse(self)

    def __str__(self):
        child_strings = [str(child) for child in self.children]
        return '(%s %s)' % (self.rule.lhs, ' '.join(child_strings))


def validate_parse(parse):
    assert isinstance(parse.rule, Rule), 'Not a Rule: %s' % parse.rule
    assert isinstance(parse.children, Iterable)
    assert len(parse.children) == len(parse.rule.rhs)
    for i in range(len(parse.rule.rhs)):
        if is_cat(parse.rule.rhs[i]):
            assert parse.rule.rhs[i] == parse.children[i].rule.lhs
        else:
            assert parse.rule.rhs[i] == parse.children[i]


def apply_semantics(rule, sems):
    # Note that this function would not be needed if we required that semantics
    # always be functions, never bare values.  That is, if instead of
    # Rule('$E', 'one', 1) we required Rule('$E', 'one', lambda sems: 1).
    # But that would be cumbersome.
    if isinstance(rule.sem, FunctionType):
        return rule.sem(sems)
    else:
        return rule.sem


def compute_semantics(parse):
    if is_lexical(parse.rule):
        return parse.rule.sem
    else:
        child_semantics = [child.semantics for child in parse.children]
        return apply_semantics(parse.rule, child_semantics)


def parse_to_pretty_string(parse, indent=0, show_sem=False):
    def indent_string(level):
        return '  ' * level

    def label(parse):
        if show_sem:
            return '(%s %s)' % (parse.rule.lhs, parse.semantics)
        else:
            return parse.rule.lhs

    def to_oneline_string(parse):
        if isinstance(parse, Parse):
          child_strings = [to_oneline_string(child) for child in parse.children]
          return '[%s %s]' % (label(parse), ' '.join(child_strings))
        else:
            return str(parse)

    def helper(parse, level, output):
        line = indent_string(level) + to_oneline_string(parse)
        if len(line) <= 100:
            print(line, file=output)
        elif isinstance(parse, Parse):
            print(indent_string(level) + '[' + label(parse), file=output)
            for child in parse.children:
                helper(child, level + 1, output)
            # TODO: Put closing parens to end of previous line, not dangling alone.
            print(indent_string(level) + ']', file=output)
        else:
            print(indent_string(level) + parse, file=output)
    output = StringIO()
    helper(parse, indent, output)
    return output.getvalue()[:-1]  # trim final newline
