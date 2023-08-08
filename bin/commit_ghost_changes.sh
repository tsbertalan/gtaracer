# For all the files git thinks are changed,
# if `git diff` actually lists no visible changes
# (because of motherfucking CRLF bullshit),
# stage them.

IFS='
'
for file in $(git ls-files --modified); do
    changes=$(git diff "$file")
    # Check for empty changes (single line ending WHATEVER THAT MEANS)
    if [[ -z "${changes// /}" ]]; then
        git add "$file"
    fi
done
