import abc
import six
import torch

@six.add_metaclass(abc.ABCMeta)
class Metric(object):
    r"""
    Base class for metric, encapsulates metric logic and APIs
    Usage:

        .. code-block:: text

            m = SomeMetric()
            for prediction, label in ...:
                m.update(prediction, label)
            m.accumulate()

    Advanced usage for :code:`compute`:

    Metric calculation can be accelerated by calculating metric states
    from model outputs and labels by build-in operators not by Python/NumPy
    in :code:`compute`, metric states will be fetched as NumPy array and
    call :code:`update` with states in NumPy format.
    Metric calculated as follows (operations in Model and Metric are
    indicated with curly brackets, while data nodes not):

        .. code-block:: text

                 inputs & labels              || ------------------
                       |                      ||
                    {model}                   ||
                       |                      ||
                outputs & labels              ||
                       |                      ||    tensor data
                {Metric.compute}              ||
                       |                      ||
              metric states(tensor)           ||
                       |                      ||
                {fetch as numpy}              || ------------------
                       |                      ||
              metric states(numpy)            ||    numpy data
                       |                      ||
                {Metric.update}               \/ ------------------

    Examples:

        For :code:`Accuracy` metric, which takes :code:`pred` and :code:`label`
        as inputs, we can calculate the correct prediction matrix between
        :code:`pred` and :code:`label` in :code:`compute`.
        For examples, prediction results contains 10 classes, while :code:`pred`
        shape is [N, 10], :code:`label` shape is [N, 1], N is mini-batch size,
        and we only need to calculate accurary of top-1 and top-5, we could
        calculate the correct prediction matrix of the top-5 scores of the
        prediction of each sample like follows, while the correct prediction
        matrix shape is [N, 5].

          .. code-block:: text

              def compute(pred, label):
                  # sort prediction and slice the top-5 scores
                  pred = paddle.argsort(pred, descending=True)[:, :5]
                  # calculate whether the predictions are correct
                  correct = pred == label
                  return paddle.cast(correct, dtype='float32')

        With the :code:`compute`, we split some calculations to OPs (which
        may run on GPU devices, will be faster), and only fetch 1 tensor with
        shape as [N, 5] instead of 2 tensors with shapes as [N, 10] and [N, 1].
        :code:`update` can be define as follows:

          .. code-block:: text

              def update(self, correct):
                  accs = []
                  for i, k in enumerate(self.topk):
                      num_corrects = correct[:, :k].sum()
                      num_samples = len(correct)
                      accs.append(float(num_corrects) / num_samples)
                      self.total[i] += num_corrects
                      self.count[i] += num_samples
                  return accs
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset states and result
        """
        raise NotImplementedError("function 'reset' not implemented in {}.".
                                  format(self.__class__.__name__))

    @abc.abstractmethod
    def update(self, *args):
        """
        Update states for metric

        Inputs of :code:`update` is the outputs of :code:`Metric.compute`,
        if :code:`compute` is not defined, the inputs of :code:`update`
        will be flatten arguments of **output** of mode and **label** from data:
        :code:`update(output1, output2, ..., label1, label2,...)`

        see :code:`Metric.compute`
        """
        raise NotImplementedError("function 'update' not implemented in {}.".
                                  format(self.__class__.__name__))

    @abc.abstractmethod
    def accumulate(self):
        """
        Accumulates statistics, computes and returns the metric value
        """
        raise NotImplementedError(
            "function 'accumulate' not implemented in {}.".format(
                self.__class__.__name__))

    @abc.abstractmethod
    def name(self):
        """
        Returns metric name
        """
        raise NotImplementedError("function 'name' not implemented in {}.".
                                  format(self.__class__.__name__))

    def compute(self, *args):
        """
        This API is advanced usage to accelerate metric calculating, calulations
        from outputs of model to the states which should be updated by Metric can
        be defined here, where Paddle OPs is also supported. Outputs of this API
        will be the inputs of "Metric.update".

        If :code:`compute` is defined, it will be called with **outputs**
        of model and **labels** from data as arguments, all outputs and labels
        will be concatenated and flatten and each filed as a separate argument
        as follows:
        :code:`compute(output1, output2, ..., label1, label2,...)`

        If :code:`compute` is not defined, default behaviour is to pass
        input to output, so output format will be:
        :code:`return output1, output2, ..., label1, label2,...`

        see :code:`Metric.update`
        """
        return args


class SpanEvaluator(Metric):
    """
    SpanEvaluator computes the precision, recall and F1-score for span detection.
    """

    def __init__(self):
        super(SpanEvaluator, self).__init__()
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

        self.num_infer_ner = 0
        self.num_label_ner = 0
        self.num_correct_ner = 0

        self.num_infer_bio = 0
        self.num_label_bio = 0
        self.num_correct_bio = 0
        
        self.predicted_spans = []

    def compute(self, probability, labels, relations):
        
        """
        Computes the precision, recall and F1-score for span detection.
        """
        

        num_correct_spans = torch.logical_and(labels == probability.argmax(-1), probability.argmax(-1) != 0).sum().item()
        
        num_infer_spans = (probability.argmax(-1) != 0).sum().item()
        # print(relations)
        num_label_spans = sum(len(rels) for rels in relations)

        return num_correct_spans, num_infer_spans, num_label_spans

    def compute_ner(self, probability, labels, relations):
        
        """
        Computes the precision, recall and F1-score for span detection.
        """
        # print(labels)
        # print(probability.argmax(-1))

        num_correct_spans = torch.logical_and(labels == probability.argmax(-1), probability.argmax(-1) != 0).sum().item()
        
        num_infer_spans = (probability.argmax(-1) != 0).sum().item()
        num_label_spans = sum(relations != 0)

        return num_correct_spans, num_infer_spans, num_label_spans

    def compute_bio(self, probability, labels, relations):
        
        """
        Computes the precision, recall and F1-score for span detection.
        """
        

        num_correct_spans = torch.logical_and(labels == probability.argmax(-1), probability.argmax(-1) != 0).sum().item()
        
        num_infer_spans = (probability.argmax(-1) != 0).sum().item()
        num_label_spans = sum(relations != 0)

        return num_correct_spans, num_infer_spans, num_label_spans

    def update(self, num_correct_spans, num_infer_spans, num_label_spans):
        """
        This function takes (num_infer_spans, num_label_spans, num_correct_spans) as input,
        to accumulate and update the corresponding status of the SpanEvaluator object.
        """
        self.num_infer_spans += num_infer_spans
        self.num_label_spans += num_label_spans
        self.num_correct_spans += num_correct_spans

    def update_ner(self, num_correct_spans, num_infer_spans, num_label_spans):
        """
        This function takes (num_infer_spans, num_label_spans, num_correct_spans) as input,
        to accumulate and update the corresponding status of the SpanEvaluator object.
        """
        self.num_infer_ner += num_infer_spans
        self.num_label_ner += num_label_spans
        self.num_correct_ner += num_correct_spans

    def update_bio(self, num_correct_spans, num_infer_spans, num_label_spans):
        """
        This function takes (num_infer_spans, num_label_spans, num_correct_spans) as input,
        to accumulate and update the corresponding status of the SpanEvaluator object.
        """
        self.num_infer_bio += num_infer_spans
        self.num_label_bio += num_label_spans
        self.num_correct_bio += num_correct_spans

    def accumulate(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns:
            tuple: Returns tuple (`precision, recall, f1 score`).
        """

        precision = float(self.num_correct_spans /
                          self.num_infer_spans) if self.num_infer_spans else 0.
        recall = float(self.num_correct_spans /
                       self.num_label_spans) if self.num_label_spans else 0.
        f1_score = float(2 * precision * recall /
                         (precision + recall)) if self.num_correct_spans else 0.
        return precision, recall, f1_score

    def accumulate_ner(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns:
            tuple: Returns tuple (`precision, recall, f1 score`).
        """
        precision = float(self.num_correct_ner /
                          self.num_infer_ner) if self.num_infer_ner else 0.
        recall = float(self.num_correct_ner /
                       self.num_label_ner) if self.num_label_ner else 0.
        f1_score = float(2 * precision * recall /
                         (precision + recall)) if self.num_correct_ner else 0.
        return precision, recall, f1_score

    def accumulate_bio(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns:
            tuple: Returns tuple (`precision, recall, f1 score`).
        """

        precision = float(self.num_correct_bio /
                          self.num_infer_bio) if self.num_infer_bio else 0.
        recall = float(self.num_correct_bio /
                       self.num_label_bio) if self.num_label_bio else 0.
        f1_score = float(2 * precision * recall /
                         (precision + recall)) if self.num_correct_bio else 0.
        return precision, recall, f1_score

    def reset(self):
        """
        Reset function empties the evaluation memory for previous mini-batches.
        """
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

        self.num_infer_ner = 0
        self.num_label_ner = 0
        self.num_correct_ner = 0

        self.num_infer_bio = 0
        self.num_label_bio = 0
        self.num_correct_bio = 0

    def name(self):
        """
        Return name of metric instance.
        """
        return "precision", "recall", "f1"
