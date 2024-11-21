tail -n 20 $1 | grep "|acc" | grep -v norm |  cut -d"|" -f8 | awk '{ total += $1 } END { print total/NR } '
