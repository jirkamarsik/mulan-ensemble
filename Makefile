JFLAGS = -g
JC = javac

.SUFFIXES: .java .class

.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	mulan/classifier/ensemble/MultiLabelEnsembleLearner.java \
	mulan/classifier/ensemble/IntersectionEnsemble.java \
	mulan/classifier/ensemble/UnionEnsemble.java \
	mulan/classifier/ensemble/MajorityVoteEnsemble.java

default: classes

classes: $(CLASSES:.java=.class)

clean:
	$(RM) $(CLASSES:.java=.class)
