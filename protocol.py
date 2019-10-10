import sys
import feature_init as f
import neighbouring_matrix_model as n
import matrices_init as m
import sequence_vector_model as s

n1 = int(sys.argv[1])
n2 = int(sys.argv[2])

f.init(64, 'data_train(full)', 0, n1)
n.model(64, 'data_train(full)', n1, n2)
m.init(64, 'data_train(full)', n1, n2)
s.model()