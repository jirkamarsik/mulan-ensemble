package mulan.experiments;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.ensemble.*;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.meta.HOMER;
import mulan.classifier.meta.HierarchyBuilder.Method;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.lazy.IBLR_ML;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.SMO;
import weka.core.Utils;

public class SIGIR11Playground {

  private static void evaluateLearner(Evaluator eval, MultiLabelInstances dataset,
                                      MultiLabelLearner learner, int numFolds,
                                      String header) {
    System.out.println(header);
    System.out.println(eval.crossValidate(learner, dataset, numFolds));
    System.out.println("\n\n");
  }

  public static void main(String[] args) throws Exception {

    String arffFilename = Utils.getOption("arff", args);
    String xmlFilename = Utils.getOption("xml", args);
    MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);

    RAkEL rakel = new RAkEL(new LabelPowerset(new J48()));
    CalibratedLabelRanking clr = new CalibratedLabelRanking(new SMO());
    MLkNN mlknn = new MLkNN();
    HOMER homer = new HOMER(new BinaryRelevance(new SMO()), 2, Method.BalancedClustering);
    IBLR_ML iblr = new IBLR_ML();

    IntersectionEnsemble intersectionEnsemble = new IntersectionEnsemble
      (new MultiLabelLearner[] {rakel.makeCopy(), clr.makeCopy(), mlknn.makeCopy(),
                                homer.makeCopy(), iblr.makeCopy()});
    UnionEnsemble unionEnsemble = new UnionEnsemble
      (new MultiLabelLearner[] {rakel.makeCopy(), clr.makeCopy(), mlknn.makeCopy(),
                                homer.makeCopy(), iblr.makeCopy()});
    MajorityVoteEnsemble majorityVoteEnsemble = new MajorityVoteEnsemble
      (new MultiLabelLearner[] {rakel.makeCopy(), clr.makeCopy(), mlknn.makeCopy(),
                                homer.makeCopy(), iblr.makeCopy()});
    MinimumEnsemble minimumEnsemble = new MinimumEnsemble
      (new MultiLabelLearner[] {rakel.makeCopy(), clr.makeCopy(), mlknn.makeCopy(),
                                homer.makeCopy(), iblr.makeCopy()});
    MaximumEnsemble maximumEnsemble = new MaximumEnsemble
      (new MultiLabelLearner[] {rakel.makeCopy(), clr.makeCopy(), mlknn.makeCopy(),
                                homer.makeCopy(), iblr.makeCopy()});
    MeanEnsemble meanEnsemble = new MeanEnsemble
      (new MultiLabelLearner[] {rakel.makeCopy(), clr.makeCopy(), mlknn.makeCopy(),
                                homer.makeCopy(), iblr.makeCopy()});
    TopKEnsemble TopKEnsemble = new TopKEnsemble
      (new MultiLabelLearner[] {rakel.makeCopy(), clr.makeCopy(), mlknn.makeCopy(),
                                homer.makeCopy(), iblr.makeCopy()});


    int numFolds = 10;
    Evaluator eval = new Evaluator();

    evaluateLearner(eval, dataset, rakel, numFolds, "RAkEL:");
    evaluateLearner(eval, dataset, clr, numFolds, "CLR:");
    evaluateLearner(eval, dataset, mlknn, numFolds, "ML-kNN:");
    evaluateLearner(eval, dataset, homer, numFolds, "HOMER:");
    evaluateLearner(eval, dataset, iblr, numFolds, "IBLR:");

    evaluateLearner(eval, dataset, intersectionEnsemble, numFolds, "Intersection rule:");
    evaluateLearner(eval, dataset, unionEnsemble, numFolds, "Union rule:");
    evaluateLearner(eval, dataset, majorityVoteEnsemble, numFolds, "Majority Vote rule:");
    evaluateLearner(eval, dataset, minimumEnsemble, numFolds, "Minimum rule:");
    evaluateLearner(eval, dataset, maximumEnsemble, numFolds, "Maximum rule:");
    evaluateLearner(eval, dataset, meanEnsemble, numFolds, "Mean rule:");
    evaluateLearner(eval, dataset, TopKEnsemble, numFolds, "Top-K rule:");
  }
}
