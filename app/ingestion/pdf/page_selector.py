def parse_exclusions(exclusion_str: str) -> set[int]:
    excluded = set()

    for part in exclusion_str.split(","):
        part = part.strip()

        if "-" in part:
            start, end = map(int, part.split("-"))
            excluded.update(range(start, end + 1))
        else:
            excluded.add(int(part))

    return excluded

def iter_pages(config):
    for page in range(config.start_page, config.end_page + 1):
        if page not in config.excluded_pages:
            yield page