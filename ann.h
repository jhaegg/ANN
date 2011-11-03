#ifndef _ANN_H_
#define _ANN_H_

typedef struct ann ann;

ann *newANN(int inputs, double threshold, int layers, ...);

ann *loadANN(const char filename[]);

int saveANN(ann *network, const char filename[]);

int destroyANN(ann *network);

int annTrain(ann *network, int samples, double input[], double expected[], double *error);

int annValidate(ann *network, int samples, double input[], double expected[], double *error);

int annProcess(ann *network, int samples, double input[], double output[]);

#endif /* _ANN_H_ */