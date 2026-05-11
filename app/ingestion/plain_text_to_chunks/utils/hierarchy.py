from app.core.utils import find_positions

def prepend_hierarchy(text: str, hierarchy: list[str]):
        headings_to_prepend = []
        last_h_level = float('inf')
        for heading in reversed(hierarchy):
            h_level = int(heading.split(':')[0][1])
            if h_level < last_h_level:
                pos_heading = find_positions(text, pattern=heading, find_first=True)
                if pos_heading is None:
                    # If heading is not found in the text, add it in hierarchy
                    last_h_level = h_level
                    headings_to_prepend.insert(0, heading)
                elif pos_heading == 0:
                    # If heading is found in the beginning, then same h_level should be ignored in subsequent iterations
                    last_h_level = h_level
        final_text = "\n".join(headings_to_prepend)
        if final_text:
            final_text += "\n\n" + text.strip()
        else:
            final_text = text.strip()
        return final_text