def make_story_mode(text: str) -> str:
    """
    Make text more narration-friendly for TTS:
    - Turns bullet/numbered list items into sentences
    - Softens headings like 'Key points:' into a spoken lead-in
    - Reduces choppy line breaks (keeps paragraph breaks)
    """
    if nottt text:
        return ""

    lines = text.splitlines()
    out: List[str] = []
    paragraph_break = False

    bullet_re = re.compile(r"^\s*(?:[-•*–—]|(\(?\d+\)?[.)]))\s+(.*)$")
    heading_re = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 \-/]{2,60})\s*:\s*$")

    for raw in lines:
        line = (raw or "").strip()

        # Preserve paragraph breaks
        if not line:
            if out and out[-1] != "\n\n":
                out.append("\n\n")
            paragraph_break = True
            continue

        # Heading-like lines ending with ":" -> lead-in sentence
        hm = heading_re.match(line)
        if hm:
            title = hm.group(1).strip()
            out.append(f"{title}. ")
            paragraph_break = False
            continue

        # Bullet / numbered list item -> sentence
        bm = bullet_re.match(line)
        if bm:
            item = (bm.group(2) or "").strip()
            if item:
                # Ensure it ends like a sentence
                if not re.search(r"[.!?]$", item):
                    item += "."
                out.append(item + " ")
            paragraph_break = False
            continue

        # Regular line
        # If a new paragraph just ended, keep a tiny spoken reset
        if paragraph_break and out and out[-1] == "\n\n":
            out.append("")
        paragraph_break = False

        # If line doesn’t end with punctuation, add a soft boundary when it was probably a wrapped line
        out.append(line + " ")

    s = "".join(out)

    # Normalize whitespace but preserve double newlines
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n\n *", "\n\n", s)
    s = s.strip()

    return s
