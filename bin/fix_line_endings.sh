# find all files with the indicate extensions
# and run fromdos on them. Then, git commit them.
IFS='
'
for ext in txt py sln user vcxproj md ahk log filters user; do
    for file in $(find . -name "*.$ext"); do
        echo "Converting $file"
        fromdos "$file"
        #git add $file
    done
done

