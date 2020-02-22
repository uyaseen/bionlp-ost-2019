from utils.pharmaco_eval.classes import BratAnnotation, NER_Evaluation
from collections import defaultdict
import os
import argparse


class evaluation(object):

    def __init__(self):
        self.format = None
        self.subtrack = []
        self.system = ""
        self.gs = ""
        self.subtask1 = ""
        self.subtask2 = ""

    def get_document_dict_by_system_id(self, system_dirs, annotation_format):
        """Takes a list of directories and returns annotations. """

        documents = defaultdict(lambda: defaultdict(int))

        for fn in os.listdir(system_dirs):
            if fn.endswith(".ann"):
                sa = annotation_format(os.path.join(system_dirs, fn))
                documents[sa.sys_id][sa.id] = sa

        return documents

    def subtracking(self, system):
        # for ta in os.listdir(system):
        #     if ta.endswith("subtask1"):
        self.subtask1 = "subtask1"
        self.subtrack.append(NER_Evaluation)
        self.format = BratAnnotation

    def checking(self,gs):
        for st in self.subtrack:
            subtask = os.path.join(self.system, "subtask1")
            for filename in os.listdir(gs):
                if filename.endswith(".ann"):
                    result = os.path.isfile(os.path.join(subtask,filename))
                    if result == False:
                        return result

        return True

    def eval(self, system, gs, output, return_score=True):
        """Evaluate the system by calling either NER_evaluation or Indexing_Evaluation.
        'system' can be a list containing either one file,  or one or more
        directories. 'gs' can be a file or a directory. """

        gold_ann = {}
        evaluations = []
        eval_score = {'p': 0., 'r': 0., 'f1': 0.}
        # system = os.path.join(input,'res')
        # gs = os.path.join(input, 'ref')

        # Handle if two files were passed on the command line
        if os.path.isdir(system) and os.path.isdir(gs):

            self.subtracking(system)

            results = []
            if not os.path.exists(output):
                os.makedirs(output)
            result_file = os.path.join(output, "scores.txt")
            file_W = open(result_file, 'w+')

            # correctFile = self.checking(gs)
            correctFile = True

            if correctFile:
                if len(self.subtrack) >= 1:
                    for st in self.subtrack:
                        # subtask = os.path.join(gs, "subtask1")
                        subtask = gs

                        for filename in os.listdir(subtask):
                            if filename.endswith(".ann"):
                                format = BratAnnotation
                                annotations = format(os.path.join(subtask, filename), gold=True)
                                gold_ann[annotations.id] = annotations

                        # subtask = os.path.join(system, "subtask1")
                        subtask = system

                        for system_id, system_ann in sorted(
                                self.get_document_dict_by_system_id(subtask, BratAnnotation).items()):
                            e = st(system_ann, gold_ann)
                            eval_score = e.print_report(file_W)
                            evaluations.append(e)
                else:
                    print("You did not follow the submission structure\n")
                    file_W.write("You did not follow the submission structure\n")
                    file_W.write("F1 : {}\n".format("ERROR"))
                    file_W.write("Precision : {}\n".format("ERROR"))
                    file_W.write("Recall : {}\n".format("ERROR"))

            else:
                print("You did not annotate all data\n")
                file_W.write("You did not follow the submission structure\n")
                file_W.write("F1 : {}\n".format("ERROR"))
                file_W.write("Precision : {}\n".format("ERROR"))
                file_W.write("Recall : {}\n".format("ERROR"))

            file_W.close()

        else:
            Exception("Must pass directory/"
                      "on command line!")

        if return_score:
            return eval_score
        return evaluations[0] if len(evaluations) == 1 else evaluations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for the PharmaCoNER track.")

    parser.add_argument("input_dir",
                        help="Directory to load GS and Submitions")
    parser.add_argument("output_dir",
                        help="Directory to print results")

    args = parser.parse_args()
    system_dir = 'input/res/subtask1/'
    gs = 'input/ref/subtask1/'
    output_dir = 'input/output'
    x = evaluation()
    x.eval(system_dir, gs, output_dir)
