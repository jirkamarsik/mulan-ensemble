package mulan.classifier.ensemble;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;

public class IntersectionEnsemble extends MultiLabelEnsembleLearner {

  public IntersectionEnsemble(MultiLabelLearner[] theClassifiers) {
    super(theClassifiers);
  }


  public MultiLabelOutput makePredictionInternal
                              (MultiLabelOutput[] classifierOutputs) {

    boolean[][] bipartitionVectors = new boolean[classifiers.length][];
    for (int i = 0; i < classifierOutputs.length; i++)
      bipartitionVectors[i] = classifierOutputs[i].getBipartition();

    boolean[] intersectionVector = new boolean[bipartitionVectors[0].length];
    for (int i = 0; i < intersectionVector.length; i++) {
      intersectionVector[i] = true;
      for (int j = 0; j < bipartitionVectors.length; j++)
        if (!bipartitionVectors[j][i])
          intersectionVector[i] = false;
    }

    return new MultiLabelOutput(intersectionVector);
  }
}
