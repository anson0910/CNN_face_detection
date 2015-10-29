# load and open files to read and write
write_file_name = 'FDDB-fold.txt'
write_file = open(write_file_name, "w")

for current_file in range(1, 11):
    print 'Processing file ' + str(current_file) + ' ...'

    read_file_name = 'FDDB-fold-' + str(current_file).zfill(2) + '.txt'

    with open(read_file_name, "r") as ins:
        array = []
        for line in ins:
            array.append(line)      # list of strings

    number_of_images = len(array)

    # write to file
    for current_image in range(number_of_images):
        write_file.write(array[current_image])

write_file.close()
