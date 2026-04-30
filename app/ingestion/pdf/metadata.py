def get_metadata(page: int, rules):
    for rule in rules:
        if rule.start <= page <= rule.end:
            return rule.metadata
    return {}