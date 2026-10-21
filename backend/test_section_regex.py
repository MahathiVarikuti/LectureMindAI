#!/usr/bin/env python3
"""Test the section detection regex."""

import re

# Current (broken) regex
CURRENT_RE = re.compile(
    r"^(name|education|experience|skills|projects|certifications|awards|certification|summary|contact|languages|coursework|interests|achievements)\s*:\s*.*$",
    re.IGNORECASE,
)

# Test against actual resume headers
test_lines = [
    "VARIKUTI MAHATHI REDDY",
    "EDUCATION",
    "Manipal Institute of Technology, Bengaluru",
    "EXPERIENCE",
    "PROJECTS",
    "TECHNICAL SKILLS",
    "CERTIFICATIONS",
    "LEADERSHIP & ACTIVITIES",
    "education:",
    "skills: python",
]

print("Current Regex Results:")
for line in test_lines:
    match = CURRENT_RE.match(line.strip())
    print(f"  {line:50} -> {bool(match)}")

# Fixed regex - match section headers without colons
FIXED_RE = re.compile(
    r"^(education|experience|projects|certifications|technical\s+skills|leadership\s+&\s+activities|languages|awards)\s*(?::|$)",
    re.IGNORECASE,
)

print("\nFixed Regex Results:")
for line in test_lines:
    match = FIXED_RE.match(line.strip())
    print(f"  {line:50} -> {bool(match)}")
