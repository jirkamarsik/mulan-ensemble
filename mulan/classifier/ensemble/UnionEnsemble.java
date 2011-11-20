package mulan.classifier.ensemble;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;

public class UnionEnsemble extends MultiLabelEnsembleLearner {

  public UnionEnsemble(MultiLabelLearner[] theClassifiers) {
    super(theClassifiers);
  }


  public MultiLabelOutput makePredictionInternal
                              (MultiLabelOutput[] classifierOutputs) {

    boolean[][] bipartitionVectors = new boolean[classifiers.length][];
    for (int i = 0; i < classifierOutputs.length; i++)
      bipartitionVectors[i] = classifierOutputs[i].getBipartition();

    boolean[] unionVector = new boolean[bipartitionVectors[0].length];
    for (int i = 0; i < unionVector.length; i++) {
      unionVector[i] = false;
      for (int j = 0; j < bipartitionVectors.length; j++)
        if (bipartitionVectors[j][i]) {
          unionVector[i] = true;
          break;
        }
    }

    return new MultiLabelOutput(unionVector);
  }
}
