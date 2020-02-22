import os


class Annotation(object):

    def __init__(self, file_name=None, root="root"):
        self.doc_id = ''
        self.sys_id = ''
        self.text = None
        self.root = root
        self.ner = []
        self.indexes = []
        self.verbose = False

        if file_name:
            self.sys_id = os.path.basename(os.path.dirname(file_name))
            self.doc_id = os.path.splitext(os.path.basename(file_name))[0]
        else:
            self.doc_id = None

    @property
    def id(self):
        return self.doc_id


class BratAnnotation(Annotation):

    def __init__(self, file_name=None, root="root", gold=True):
        self.doc_id = ''
        self.sys_id = ''
        self.root = root
        self.ner = []
        self.verbose = False

        if file_name:
            self.sys_id = os.path.basename(os.path.dirname(file_name))
            self.doc_id = os.path.splitext(os.path.basename(file_name))[0]

            self.parse_tags(file_name, gold)
            self.file_name = file_name
        else:
            self.doc_id = None

    def get_ner(self):
        return self.ner

    def parse_tags(self, file_name=None, gold=None):
        if file_name is not None:
            for row in open(file_name, 'r', encoding='utf-8'):
                line = row.strip()
                if line.startswith("T"):  # Lines is a Brat TAG
                    try:
                        label = line.split("\t")[1].split()
                        tag = label[0]
                        if tag == "UNCLEAR" and gold == True:
                            if self.verbose:
                                print("\tSkipping line (comment):\t" + line)
                            continue

                        start = int(label[1])
                        end = int(label[2])
                        self.ner.append((tag, start, end))
                    except IndexError:
                        print("ERROR! Index error while splitting sentence '" +
                              line + "' in document '" + file_name + "'!")
                else:  # Line is a Brat comment
                    if self.verbose:
                        print("\tSkipping line (comment):\t" + line)


class IndexingAnnotation(Annotation):
    """ This class models the PharmaCoNER TSV annotation format."""
    def __init__(self, file_name=None, root="root"):
        self.doc_id = ''
        self.sys_id = ''
        self.root = root
        self.indexes = []
        self.verbose = False

        if file_name:
            self.sys_id = os.path.basename(os.path.dirname(file_name))
            self.doc_id = os.path.splitext(os.path.basename(file_name))[0]

            self.parse_indexes(file_name)
            self.file_name = file_name
        else:
            self.doc_id = None

    def get_index(self):
        return self.indexes

    def parse_indexes(self, file_name=None):
        if file_name is not None:
            for row in open(file_name, 'r'):
                line = row.strip()
                sys_id = line.split("\t")[0]
                sys_code = line.split("\t")[1]
                if sys_id == self.doc_id:
                    self.indexes.append(sys_code)
                else:
                    self.indexes.append(sys_code)
                    print("WARNING: Filename '" + self.doc_id + "' and File ID '" + sys_id +"' does not match")


class Evaluate(object):
    """Base class with all methods to evaluate the different subtracks."""

    def __init__(self, sys_ann, gs_ann):
        self.tp = []
        self.fp = []
        self.fn = []
        self.doc_ids = []
        self.verbose = False

        self.sys_id = sys_ann[list(sys_ann.keys())[0]].sys_id

    @staticmethod
    def get_tagset_ner(annotation):
        return annotation.get_ner()

    @staticmethod
    def get_tagset_indexes(annotation):
        return annotation.get_index()
    @staticmethod
    def recall(tp, fn):
        try:
            return len(tp) / float(len(fn) + len(tp))
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def precision(tp, fp):
        try:
            return len(tp) / float(len(fp) + len(tp))
        except ZeroDivisionError:
            return 0.0

    @staticmethod
    def F_beta(p, r, beta=1):
        try:
            return (1 + beta**2) * ((p * r) / (p + r))
        except ZeroDivisionError:
            return 0.0

    def micro_recall(self):
        try:
            return sum([len(t) for t in self.tp]) /  \
                float(sum([len(t) for t in self.tp]) +
                      sum([len(t) for t in self.fn]))
        except ZeroDivisionError:
            return 0.0

    def micro_precision(self):
        try:
            return sum([len(t) for t in self.tp]) /  \
                float(sum([len(t) for t in self.tp]) +
                      sum([len(t) for t in self.fp]))
        except ZeroDivisionError:
            return 0.0

    def _print_docs(self):
        for i, doc_id in enumerate(self.doc_ids):
            mp = Evaluate.precision(self.tp[i], self.fp[i])
            mr = Evaluate.recall(self.tp[i], self.fn[i])

            str_fmt = "{:<35}{:<15}{:<20}"

            print(str_fmt.format(doc_id,
                                 "Precision",
                                 "{:.4}".format(mp)))

            print(str_fmt.format("",
                                 "Recall",
                                 "{:.4}".format(mr)))

            print(str_fmt.format("",
                                 "F1",
                                 "{:.4}".format(Evaluate.F_beta(mp, mr))))

            print("{:-<60}".format(""))

    def _print_summary(self,file_W):
        mp = self.micro_precision()
        mr = self.micro_recall()
        f1_score = Evaluate.F_beta(mr, mp)

        file_W.write("F1 : {}\n".format(f1_score))
        file_W.write("Precision : {}\n".format(mp))
        file_W.write("Recall : {}\n".format(mr))

        print("F1 : {}\n".format(f1_score))
        print("Precision : {}\n".format(mp))
        print("Recall : {}\n".format(mr))
        eval_score = {'p': mp, 'r': mr, 'f1': f1_score}
        return f1_score


    def print_docs(self):
        print("\n")
        print("Report ({}):".format(self.sys_id))
        print("{:-<60}".format(""))
        print("{:<35}{:<15}{:<20}".format("Document ID", "Measure", "Micro"))
        print("{:-<60}".format(""))
        self._print_docs()

    def print_report(self, file_W):
        return self._print_summary(file_W)


class EvaluateSubtrack1(Evaluate):
    """Class for running the NER evaluation."""

    def __init__(self, sys_sas, gs_sas):
        self.tp = []
        self.fp = []
        self.fn = []
        # self.num_sentences = []
        self.doc_ids = []
        self.verbose = False

        self.sys_id = sys_sas[list(sys_sas.keys())[0]].sys_id
        self.label = "Subtrack 1 [NER]"

        for doc_id in sorted(list(set(sys_sas.keys()) & set(gs_sas.keys()))):

            gold = set(self.get_tagset_ner(gs_sas[doc_id]))
            sys = set(self.get_tagset_ner(sys_sas[doc_id]))

            self.tp.append(gold.intersection(sys))
            self.fp.append(sys - gold)
            self.fn.append(gold - sys)
            self.doc_ids.append(doc_id)

    def _print_docs(self):
        for i, doc_id in enumerate(self.doc_ids):
            mp = EvaluateSubtrack1.precision(self.tp[i], self.fp[i])
            mr = EvaluateSubtrack1.recall(self.tp[i], self.fn[i])

            str_fmt = "{:<35}{:<15}{:<20}"

            print(str_fmt.format("",
                                 "Precision",
                                 "{:.4}".format(mp)))

            print(str_fmt.format("",
                                 "Recall",
                                 "{:.4}".format(mr)))

            print(str_fmt.format("",
                                 "F1",
                                 "{:.4}".format(Evaluate.F_beta(mp, mr))))

            print("{:-<60}".format(""))

    def _print_summary(self, file_W):
        mp = self.micro_precision()
        mr = self.micro_recall()
        f1_score = Evaluate.F_beta(mr, mp)
        # ml = self.micro_leak()

        file_W.write("F1 : {}\n".format(f1_score))
        file_W.write("Precision : {}\n".format(mp))
        file_W.write("Recall : {}\n".format(mr))

        print("F1 : {}\n".format(f1_score))
        print("Precision : {}\n".format(mp))
        print("Recall : {}\n".format(mr))
        eval_score = {'p': mp, 'r': mr, 'f1': f1_score}
        return eval_score


class EvaluateSubtrack2(Evaluate):
    """Class for running the Concept Indexing evaluation."""

    def __init__(self, sys_sas, gs_sas):
        self.tp = []
        self.fp = []
        self.fn = []
        self.doc_ids = []
        self.verbose = False

        self.sys_id = sys_sas[list(sys_sas.keys())[0]].sys_id
        self.label = "Subtrack 2 [Indexing]"

        for doc_id in sorted(list(set(sys_sas.keys()) & set(gs_sas.keys()))):

            gold = set(self.get_tagset_indexes(gs_sas[doc_id]))
            sys = set(self.get_tagset_indexes(sys_sas[doc_id]))

            self.tp.append(gold.intersection(sys))
            self.fp.append(sys - gold)
            self.fn.append(gold - sys)
            self.doc_ids.append(doc_id)


class PharmaconerEvaluation(object):
    """Base class for running the evaluations."""

    def __init__(self):
        self.evaluations = []

    def add_eval(self, e, label=""):
        e.sys_id = "SYSTEM: " + e.sys_id
        e.label = label
        self.evaluations.append(e)

    def print_docs(self):
        for e in self.evaluations:
            e.print_docs()

    def print_report(self, file_W=None):
        for e in self.evaluations:
            return e.print_report(file_W)


class NER_Evaluation(PharmaconerEvaluation):
    """Class for running the NER evaluation (Subtrack 1)."""

    def __init__(self, annotator_cas, gold_cas, **kwargs):
        self.evaluations = []

        # Basic Evaluation
        self.add_eval(EvaluateSubtrack1(annotator_cas, gold_cas, **kwargs),
                      label="SubTrack 1 [NER]")


class Indexing_Evaluation(PharmaconerEvaluation):
    """Class for running the Concept Indexing evaluation (Subtrack 2)."""

    def __init__(self, annotator_cas, gold_cas, **kwargs):
        self.evaluations = []

        self.add_eval(EvaluateSubtrack2(annotator_cas, gold_cas, **kwargs),
                      label="SubTrack 2 [Indexing]")
