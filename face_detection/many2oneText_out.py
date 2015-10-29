# load and open files to read and write
write_file_name = 'FDDB-out.txt'
write_file = open(write_file_name, "w")

for current_file in range(1, 11):
    print 'Processing file ' + str(current_file) + ' ...'

    read_file_name = 'fold-' + str(current_file).zfill(2) + '-out.txt'

    with open(read_file_name, "r") as ins:
        array = []
        for line in ins:
            array.append(line)      # list of strings

    number_of_lines = len(array)

    # write to file
    for current_line in range(number_of_lines):
        write_file.write(array[current_line])

write_file.close()
