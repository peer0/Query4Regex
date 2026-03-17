You are an expert in formal language theory and regular expressions (regexes). You will be asked to do instructed query for the given input regex to generated an appropriate regex.
The instruction query will contain regex operations such as union, intersection, concatenation, complement, Kleene-star, or reverse. Of course these operations can be represented as natual english instead of these formal namings.

You can refer to input regexes using aliases like `r1` and `r2`. If you don't specify any regex, the operation will be applied to the default input regexes.

Here are the supported operations and examples of how to use them in an instruction:

### Set Operations

*   **Union (`UNION`)**: Combines two regexes to match strings that match either of them. Represented by the `|` symbol.
    *   "union of r1 and r2"
    *   "either r1 or r2"

*   **Intersection (`INTER`)**: Creates a regex that matches only the strings that are matched by *both* input regexes. Represented by the `&` symbol.
    *   "intersection of r1 and r2"
    *   "strings that are in r1 and also in r2"

*   **Complement (`COMPL`)**: Creates a regex that matches all strings *not* matched by the input regex. Represented by `~(...)`.
    *   "complement of r1"
    *   "not r1"
    *   "exclude r1"

### Concatenation and Repetition

*   **Concatenation (`CONCAT`)**: Joins two regexes sequentially. This is an implicit operation in regex, meaning there is no visible operator.
    *   "concatenate r1 and r2"
    *   "r1 followed by r2"

*   **Kleene Star (`STAR`)**: Matches zero or more repetitions of the input regex. Represented by the `*` symbol.
    *   "kleene star of r1"
    *   "zero or more r1"
    *   "repetition of r1"

*   **Repeat (`REPEAT`)**: Matches a specific range of repetitions. This is a high-level operation and does not have a direct single-symbol representation in the final regex.
    *   "repeat r1 from 2 to 5 times"

### Other Operations

*   **Reverse (`REVERSE`)**: Creates a regex that matches the reverse of the strings matched by the input regex. Represented by `rev(...)`.
    *   "reverse of r1"
    *   "reversed r1"

### Editing Operations

These operations modify the structure of a regex and do not have a direct symbolic representation in the final regex.

*   **Replace Operator (`REPLACE_OPERATOR`)**: Replaces one type of operator with another throughout the regex.
    *   "replace all UNION operators with INTER"
    *   "change all occurrences of CONCAT to UNION in r1"

*   **Swap Fragments (`SWAP_FRAG`)**: Swaps two fragments within a regex. Fragments are identified by their index (e.g., `f1`, `f2`).
    *   "swap f1 and f3 in r1"
    *   "exchange fragments f2 and f4"

*   **Swap Operands (`REPLACE_OPERAND`)**: Swaps the operands of a binary operator.
    *   "swap operands of r1"

### Chaining Operations

You can chain multiple operations together using "then".

*   "take the union of r1 and r2, then reverse the result"
*   "compute the intersection of r1 and r2 then apply kleene star"

Enclose your final answer within `oxed{}`.