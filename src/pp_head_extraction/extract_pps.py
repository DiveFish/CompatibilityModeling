import argparse
import csv


def write_pps_to_file(input_file: str, output_file: str):
    with open(input_file) as in_file, open(output_file, "w") as out_file:
        in_reader = csv.reader(in_file, delimiter="\t")
        out_writer = csv.writer(out_file, delimiter="\t")

        pp_start = 0
        pp_end = head_step_size = 7
        head_id = 5
        true_head = "1"

        for line in in_reader:
            # Add the preposition and noun inside the PP.
            pp_out = line[pp_start:pp_end]

            index = pp_end
            while len(pp_out) <= pp_end:
                if line[index + head_id] == true_head:
                    # Add the head of the PP.
                    pp_out.extend(line[index:index + head_step_size])
                index += head_step_size

            out_writer.writerow(pp_out)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("input", help="Input CoNLL or TSV file")
    argp.add_argument("output", help="Output path for the csv file of extracted PPs")    

    args = argp.parse_args()

    write_pps_to_file(args.input, args.output)
