# Sample Results

These results come from the current adaptive pipeline and character-level
reranker. They are not hardcoded labels.

| sample key | top candidate | notes |
| --- | --- | --- |
| 022214 | `静赌景注` | stable top result |
| 034905 | `星绝步放` | stable top result |
| 034918 | `狱己擦九` | high-risk; tied closely with `狱己擦力` |
| 034933 | `降离恶讯` | stable top result |
| 034944 | `寸遍惯警` | stable top result |

The most difficult case is `034918`. The correct-looking candidate improved
after adding wider dominant-color variants and stronger low-saturation dark-line
suppression. It should still be treated as ambiguous because `九` and `力`
receive very close evidence after preprocessing.
