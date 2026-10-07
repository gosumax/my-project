# Same-source numeric representation candidate

Native direct grayscale stack crops are only18 pixels high. Source-bound saved
examples T05/H5559 seat5, T09/H5800 seats3/5 and T05/H5932 seat2 have literal
numeric primary text below .95. SQLite crop representations resolve two of these
examples but do not retain three separate observation pixel records, so its later
confirmation cannot substitute for the required first numeric triple.

A bounded offline experiment used22 saved sources, five fixed representations,
and the unchanged Tiny model. The direct native representation confidently read
3/15 numeric frames. Two-times nearest-neighbor read15/15, bilinear12/15, refined
native6/15 and refined nearest9/15. No confident wrong number occurred among these
22 sources, which include7 forced/fade/blank controls. This is a development
diagnostic, not a general OCR accuracy claim or independent Smoke19 acceptance.

FULL wraps only the direct live START_STACK client. The primary request/output
remain intact. For an OK numeric primary below the unchanged .95 threshold, one
additional exact2x nearest-neighbor image is submitted synchronously to the same
worker. There is no callback queue recursion. The result is separate evidence;
it must agree exactly with the primary digits and exceed the same confidence gate.
The wrapper never selects between multiple transformations or replaces a number.

Consumption reproduces source and scaled hashes/shapes, expected fixed-seat OCR
item identity (table/frame/index), model name and weight hash, and primary response
fields. Malformed, missing, conflicting or low-confidence evidence is UNKNOWN.
The first-visible selector and independent first-visible fact verifier use the
same validator. They still require three consecutive original observations and
all source/PTS/context/current forced-overlay gates. Original grayscale/RGB and
raw primary/derived responses stay in receipts. Default OFF behavior is unchanged.

The extra exact inference interval is retained as START_STACK_SAME_SOURCE_SCALE;
aggregate request inference time includes both serial requests. An extra-worker
error preserves the primary output and marks only supplemental evidence UNKNOWN.

Targeted243tests passed. Independent read-only review found no material issue.
Official fresh acceptance and full-video quality are still required. No golden
promotion. Prior202-hash manifest and incomplete short_v5 receipts are retained.
