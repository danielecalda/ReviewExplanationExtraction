class Rule:
    """Represents a CFG rule with a semantic attachment."""

    def __init__(self, lhs, rhs, sem=None):
        self.lhs = lhs
        self.rhs = tuple(rhs.split()) if isinstance(rhs, str) else rhs
        self.sem = sem
        validate_rule(self)

    def __str__(self):
        """Returns a string representation of this Rule."""
        return 'Rule' + str((self.lhs, ' '.join(self.rhs), self.sem))

def is_cat(label):
    """
    Returns true iff the given label is a category (non-terminal), i.e., is
    marked with an initial '$'.
    """
    return label.startswith('$')

def is_lexical(rule):
    """
    Returns true iff the given Rule is a lexical rule, i.e., contains only
    words (terminals) on the RHS.
    """
    return all([not is_cat(rhsi) for rhsi in rule.rhs])

def is_unary(rule):
    """
    Returns true iff the given Rule is a unary compositional rule, i.e.,
    contains only a single category (non-terminal) on the RHS.
    """
    return len(rule.rhs) == 1 and is_cat(rule.rhs[0])

def is_binary(rule):
    """
    Returns true iff the given Rule is a binary compositional rule, i.e.,
    contains exactly two categories (non-terminals) on the RHS.
    """
    return len(rule.rhs) == 2 and is_cat(rule.rhs[0]) and is_cat(rule.rhs[1])

def validate_rule(rule):
    """Returns true iff the given Rule is well-formed."""
    assert is_cat(rule.lhs), 'Not a category: %s' % rule.lhs
    assert isinstance(rule.rhs, tuple), 'Not a tuple: %s' % rule.rhs
    for rhs_i in rule.rhs:
        assert isinstance(rhs_i, str), 'Not a string: %s' % rhs_i

def is_optional(label):
    """
    Returns true iff the given RHS item is optional, i.e., is marked with an
    initial '?'.
    """
    return label.startswith('?') and len(label) > 1

def contains_optionals(rule):
    """Returns true iff the given Rule contains any optional items on the RHS."""
    return any([is_optional(rhsi) for rhsi in rule.rhs])

def equals(rule1, rule2):
    if rule1.lhs == rule2.lhs and rule1.rhs == rule2.rhs:
        return True
    else:
        return False

def rule_not_in_list(rule, rules):
    for temp_rule in rules:
        if equals(rule, temp_rule):
            return False
    return True
