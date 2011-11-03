#include "ann.h"
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>

#define TRAINING_RATE 0.2
#define ERROR(code) { errno = code; goto failure; }
#define PROPAGATE_ERROR(code) if(code != 0) goto failure
#define ASSERT_FREE(memory) if(memory == NULL) goto end; else free(memory);

typedef struct workbuffer
{
	double **nodeWeightStart;
	double *storage;
	double *errorGradients;
} workbuffer;

struct ann
{
	int inputs;
	double threshold;
	int layers;
	int *layerNodes;
	int numWeights;
	double *weights;
	workbuffer *buffer;
};

ann *generateANN(int inputs, double threshold, int layers, int *numLayers);
int generateWorkBuffer(ann *network);
int destroyWorkBuffer(ann *network);
void nodeCalculate(workbuffer *buffer, int nodeIndex, int inputs, double threshold);
__inline double sigmoid(double t);
void backpropagate(ann *network, double *outputError);

ann *newANN(int inputs, double threshold, int layers, ...)
{
	va_list args;
	int index;
	int *layerNodes = NULL;

	/* Validate parameters. */
	if(inputs < 1 || layers < 1)
		ERROR(EDOM);

	/* Create new array to contain number of nodes per layer. */
	layerNodes = (int*)calloc(layers, sizeof(int));

	if(layerNodes == NULL)
		ERROR(ENOMEM);

	/* Read number of layers per node from variable argument list. */
	for(index = 0, va_start(args, layers); index < layers; index++)
		if((layerNodes[index] = va_arg(args, int)) < 1)
			ERROR(EDOM);

	return generateANN(inputs, threshold, layers, layerNodes);	
	
failure:
	ASSERT_FREE(layerNodes);
end:
	return NULL;
}

ann *loadANN(const char filename[])
{
	FILE *fileHandle = NULL;
	int inputs, layers;
	double threshold;
	int *layerNodes = NULL;	
	ann *network;

	/* Open file and read all parameters */
	PROPAGATE_ERROR(fopen_s(&fileHandle, filename, "rb"));

	if(fread(&inputs, sizeof(int), 1, fileHandle) != sizeof(int))
		ERROR(errno);

	if(fread(&threshold, sizeof(double), 1, fileHandle) != sizeof(double))
		ERROR(errno);

	if(fread(&layers, sizeof(int), 1, fileHandle) != sizeof(int))
		ERROR(errno);

	if(inputs < 1 || layers < 1)
		ERROR(EDOM);

	layerNodes = (int*)calloc(layers, sizeof(int));

	if(fread(layerNodes, sizeof(int), layers, fileHandle) != sizeof(int) * layers)
		ERROR(errno);

	/* Generate a new neural network */
	network = generateANN(inputs, threshold, layers, layerNodes);

	if(network == NULL)
		ERROR(errno);

	/* Read stored weights. */
	if(fread(network->weights, sizeof(double), network->numWeights, fileHandle) != sizeof(double) * network->numWeights)
		ERROR(errno);

	fclose(fileHandle);
	return network;

failure:
	if(fileHandle != NULL)
		fclose(fileHandle);

	destroyANN(network);

	return NULL;
}

int saveANN(ann *network, const char filename[])
{
	FILE *fileHandle = NULL;

	PROPAGATE_ERROR(fopen_s(&fileHandle, filename, "rb"));

	if(fwrite(&network->inputs, sizeof(int), 1, fileHandle) != sizeof(int))
		ERROR(errno);

	if(fwrite(&network->threshold, sizeof(double), 1, fileHandle) != sizeof(double))
		ERROR(errno);

	if(fwrite(&network->layers, sizeof(int), 1, fileHandle) != sizeof(int))
		ERROR(errno);

	if(fwrite(network->layerNodes, sizeof(int), network->layers, fileHandle) != sizeof(int) * network->layers)
		ERROR(errno);

	if(fwrite(network->weights, sizeof(double), network->numWeights, fileHandle) != sizeof(double) * network->numWeights)
		ERROR(errno);

	fclose(fileHandle);

	return 1;

failure:
	if(fileHandle != NULL)
		fclose(fileHandle);
	return 0;
}

int destroyANN(ann *network)
{
	if(network == NULL)
		return 0;

	ASSERT_FREE(network->layerNodes);
	ASSERT_FREE(network->weights);

	destroyWorkBuffer(network);

end:
	free(network);
	return 1;
}

int annTrain(ann *network, int samples, double input[], double expected[], double *error)
{
	int sampleIndex, index;
	double *output, *outputError;
	const int outputCount = network->layerNodes[network->layers - 1];

	/* Initialize storage */
	if(error == NULL)
		ERROR(EDOM);

	output = (double*)calloc(outputCount, sizeof(double));

	if(output == NULL)
		ERROR(ENOMEM);

	outputError = (double*)calloc(outputCount, sizeof(double));

	if(outputError == NULL)
		ERROR(ENOMEM);

	/* Train using each provided sample */
	for(sampleIndex = 0; *error = 0, sampleIndex < samples; sampleIndex++)
	{
		if(!annProcess(network, 1, &input[sampleIndex * network->inputs], output))
			ERROR(errno);

		for(index = 0; index < outputCount; index++)
		{
			outputError[index] = output[index] - expected[sampleIndex * outputCount + index];
			*error += outputError[index] * outputError[index];
		}

		backpropagate(network, outputError);
	}

	free(outputError);
	free(output);

	return 1;
failure:
	ASSERT_FREE(output);
	ASSERT_FREE(outputError);
end:
	return 0;
}

int annValidate(ann *network, int samples, double input[], double expected[], double *error)
{
	int index;
	double *output = NULL;
	const int outputCount = network->layerNodes[network->layers - 1] * samples;

	/* Initialize Storage */
	if(error == NULL)
		ERROR(EDOM);

	output = (double*)calloc(outputCount, sizeof(double));

	if(output == NULL)
		ERROR(ENOMEM);

	/* Process samples and sum errors */
	if(!annProcess(network, samples, input, output))
		ERROR(errno);

	for(index = 0, *error = 0; index < outputCount; index++)
		*error += (output[index] - expected[index]) * (output[index] - expected[index]);

	free(output);
	return 1;

failure:
	ASSERT_FREE(output);
end:
	return 0;
}

int annProcess(ann *network, int samples, double input[], double output[])
{
	int sampleIndex, layerIndex, nodePos, nodeIndex, prevNumNodes;
	const int outputCount = network->layerNodes[network->layers - 1];

	if(network == NULL || samples < 1 || input == NULL || output == NULL)
		ERROR(EDOM);

	for(sampleIndex = 0; sampleIndex < samples; sampleIndex++)
	{
		/* Store current input sample */
		memcpy(network->buffer->storage, &input[sampleIndex * network->inputs], network->inputs * sizeof(double));
		prevNumNodes = network->inputs;

		/* Process each node layer by layer */
		for(layerIndex = 0, nodePos = network->inputs; layerIndex < network->layers; layerIndex++)
		{
			for(nodeIndex = 0; nodeIndex < network->layerNodes[layerIndex]; nodeIndex++, nodePos++)
				nodeCalculate(network->buffer, nodePos, prevNumNodes, network->threshold);
		}

		/* Store output in array */
		memcpy(&output[sampleIndex * outputCount], &network->buffer->storage[nodePos - outputCount], outputCount * sizeof(double));
	}

	return 1;
failure:
	return 0;
}

ann *generateANN(int inputs, double threshold, int layers, int *layerNodes)
{
	ann *network = NULL;
	int index, prevNumNodes;

	/* Initialize data structure */
	network = (ann*)malloc(sizeof(ann));

	if(network == NULL)
		ERROR(ENOMEM);

	network->inputs = inputs;
	network->threshold = threshold;
	network->layers = layers;	
	network->layerNodes = layerNodes;
	network->numWeights = 0;
	network->buffer = NULL;

	/* Allocate memory for weight storage and initialize layerNodes.
	 * Each node requires one weight for each input/nodes in previous layer
	 * and one weight for threshold weight. 
	 */
	for(index = 0, prevNumNodes = inputs; index < layers; index++)
	{
		network->numWeights += layerNodes[index] * (prevNumNodes + 1);
		prevNumNodes = layerNodes[index];
	}

	network->weights = (double*)calloc(network->numWeights, sizeof(double));

	if(network->weights == NULL)
		ERROR(ENOMEM);

	/* Generate work data buffer for data structure. */
	if(!generateWorkBuffer(network))
		ERROR(errno);

	return network;
	
failure:
	destroyANN(network);
	return NULL;

}

int generateWorkBuffer(ann *network)
{
	int index, prevNumNodes, weightStartIndex, weightIndex, nodeIndex, numNodes;

	/* Initialize data structure */

	network->buffer = (workbuffer*)malloc(sizeof(workbuffer));

	if(network->buffer == NULL)
		ERROR(ENOMEM);

	for(index = 0, numNodes = 0; index < network->layers; index++)
		numNodes += network->layerNodes[index];

	network->buffer->storage = (double*)calloc(network->inputs + numNodes, sizeof(double));

	if(network->buffer->storage == NULL)
		ERROR(ENOMEM);

	network->buffer->nodeWeightStart = (double**)calloc(numNodes, sizeof(double*));

	if(network->buffer->nodeWeightStart == NULL)
		ERROR(ENOMEM);

	network->buffer->errorGradients = (double*)calloc(numNodes, sizeof(double));

	if(network->buffer->errorGradients == NULL)
		ERROR(ENOMEM);

	/* Create weight storage pointers for each node starting with the first node in the
	 * first layer. These pointers should point to the first weight associated with the 
	 * node. Each node has a weight for each node in the previous layer and a threshold
	 * weight.
	 */
	for(index = 0, prevNumNodes = network->inputs, weightStartIndex = 0, weightIndex = 0; index < network->layers; index++)
	{
		for(nodeIndex = 0; nodeIndex < network->layerNodes[index]; nodeIndex++, weightStartIndex++)
		{
			network->buffer->nodeWeightStart[weightStartIndex] = &network->weights[weightIndex];
			weightIndex += prevNumNodes + 1;
		}
	}

	return 1;

failure:
	/* Calling function takes care of cleanup. */
	return 0;
}

int destroyWorkBuffer(ann *network)
{
	if(network->buffer == NULL)
		return 0;

	ASSERT_FREE(network->buffer->nodeWeightStart);
	ASSERT_FREE(network->buffer->storage);
	ASSERT_FREE(network->buffer->errorGradients);

end:		
	free(network->buffer);
	return 1;
}

void nodeCalculate(workbuffer *buffer, int nodeIndex, int inputs, double threshold)
{
	double weightedSum = 0.0;
	int index;

	for(index = 0; index < inputs; index++)
		weightedSum += buffer->storage[nodeIndex - inputs + index] * buffer->nodeWeightStart[nodeIndex][index];

	weightedSum += threshold * buffer->nodeWeightStart[nodeIndex][inputs];

	buffer->storage[nodeIndex] = sigmoid(weightedSum);
}

__inline double sigmoid(double t)
{
	return 1.0/(1.0 + exp(-t));
}

void backpropagate(ann *network, double *outputError)
{
	int index, nodePos, layerIndex, nodeIndex, weightCount, prevWeightCount;
	double error;

	for(index = 0, nodePos = 0; index < network->layers; index++)
		nodePos += network->layerNodes[index];
	
	/* Backpropagate error from last node to first node storing error gradients. */
	for(index = network->layerNodes[network->layers - 1] - 1; index >= 0 ; index--, nodePos--)
		network->buffer->errorGradients[nodePos] = network->buffer->storage[nodePos + network->inputs] * (1.0 - network->buffer->storage[nodePos + network->inputs]) * outputError[index];

	for(layerIndex = network->layers - 1; layerIndex >= 0; layerIndex--)	
	{
		for(; nodePos >= 0; nodePos--)
		{
			for(index = 0, error = 0.0; index < network->layerNodes[layerIndex]; index++)
				error += network->buffer->nodeWeightStart[nodePos + network->layerNodes[layerIndex]][index] * network->buffer->errorGradients[nodePos + index];

			network->buffer->errorGradients[nodePos] = network->buffer->storage[nodePos + network->inputs] * (1.0 - network->buffer->storage[nodePos + network->inputs]) * error;
		}
	}

	/* Update weights */
	for(layerIndex = 0, prevWeightCount = 0; layerIndex < network->layers; layerIndex++)
	{
		if(layerIndex == 0)
			weightCount = network->inputs;
		else
			weightCount = network->layerNodes[layerIndex - 1];

		for(nodeIndex = 0; nodeIndex < network->layerNodes[layerIndex]; nodeIndex++, nodePos++)
		{
			for(index = 0; index < weightCount; index++)
				network->buffer->nodeWeightStart[nodePos][index] += TRAINING_RATE * network->buffer->storage[index - prevWeightCount];
		}

		prevWeightCount = weightCount;
	}	
}