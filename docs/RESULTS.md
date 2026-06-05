# MyCaptchaOCR Sample Results

These results come from the current adaptive pipeline and character-level
reranker on `data/raw/sample-[0-9]*.png`. They are not hardcoded labels.

| sample file | top candidate | notes |
| --- | --- | --- |
| `sample-022214.png` | `静赌景注` | stable top result |
| `sample-034905.png` | `星绝步放` | stable top result |
| `sample-034918.png` | `狱己擦九` | high-risk; tied closely with `狱己擦力` |
| `sample-034933.png` | `降离恶讯` | stable top result |
| `sample-034944.png` | `寸遍惯警` | stable top result |

The most difficult case is `034918`. The correct-looking candidate improved
after adding wider dominant-color variants and stronger low-saturation dark-line
suppression. It should still be treated as ambiguous because `九` and `力`
receive very close evidence after preprocessing.
