from parsing.rule import Rule


def sems_0(sems):
    return sems[0]

def sems_1(sems):
    return sems[1]

def merge_2_dicts(d1, d2):
    if not d2:
        return [d1]
    result = [d1, d2]
    return result

rules_review = [
    Rule('$ROOT', '$Review', sems_0),
    Rule('$Review', '$ReviewParts', sems_0),
    Rule('$ReviewParts', '?$Optionals $ReviewPattern ?$Optionals ?$ReviewParts ?$Optionals', lambda sems: merge_2_dicts(sems[1], sems[3])),
    Rule('$Optionals', '$Optional ?$Optionals')
]
