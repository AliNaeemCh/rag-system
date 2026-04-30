PDF_TRANSCRIBER_PROMPT = """Instructions for extracting text from the image:

1. Text Extraction:
   - Extract **all text** from the image.
   - Keep the original text layout intact, including all new lines, paragraphs, and spacing.
   - Do NOT rephrase, summarize, or shrink the text; maintain the original wording.
   - Do NOT omit any information.

2. Handling Infographics or Tables:
   - Understand the content and represent it in **explanatory plain text**.
   - Do not dump raw table or infographic text.
   - Ensure that each sentence in the explanatory text is of reasonable length: not too long and not too short.
   - Every piece of information in an infographic is relevant and MUST be transcribed.
   - Every line item in a table is relevant and MUST be transcribed.

3. Heading Structure

   * Use format: `<H1> Title`, `<H2> Title`, `<H3> Title`, etc.

   * Headings must be short, standalone, and visually prominent labels.

   * **Hierarchy rules (strict):**

      * Preserve the original logical heading structure exactly as it appears in the source.
      * Do not reorder, flatten, normalize, or “fix” heading levels.
      * If the source hierarchy is `H1 → H2 → H2 → H3 → H3 → H2`, it must be reproduced identically.
      * Heading levels are determined by original structure, not stylistic judgment.

   * **H1 constraint:**

      * `<H1>` is only used when a new or distinct top-level topic begins.

   * **Level cues (supporting rule):**

      * Use numbering, visual prominence, and semantic meaning as hints when mapping hierarchy (e.g., 1 → H1, 1.1 → H2, 1.1.1 → H3), but never override the original structure.

   * Every heading must include a title after the tag (`<Hx> Title`).

   * If content begins without an explicit heading but introduces a new topic, generate a concise heading for it.

4. Content Continuity Check

   * You will be provided with the previous page’s transcription and heading hierarchy for context.
   * Treat this previous content as **logically preceding the current image content**.
   * The heading structure for the current page must **continue seamlessly from the previous page**, if applicable.
   * Maintain continuity in hierarchy level where the topic carries forward; do not reset structure unless a new top-level topic begins.
   * If the content is a continuation of the previous page, prepend `<CONT.>` at the very beginning of the output before starting the transcription.
   * Do not repeat anything from the previous page content.

5. Numerical Values:
   - Always include the **unit** of the number if known (e.g., rupees, %, kilograms).
   - Units must be explicitly mentioned **every time** a number appears.

6. Formatting Rules:
   - Output **plain text paragraphs only**.
   - Do NOT use lists, tables, or structured formats.

7. Exclusion Rules:
   - Financial figures that are labelled 'unconsolidated' or 'standalone' MUST be excluded.
   - Ignore headers and footers entirely."""