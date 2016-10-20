/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl{
	public ArrayList<Node> inputNodes=null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes=null;// list of the output layer nodes
	
	public ArrayList<Instance> trainingSet=null;//the training set
	
	Double learningRate=1.0; // variable to store the learning rate
	int maxEpoch=1; // variable to store the maximum number of epochs
	
	/**
 	* This constructor creates the nodes necessary for the neural network
 	* Also connects the nodes of different layers
 	* After calling the constructor the last node of both inputNodes and  
 	* hiddenNodes will be bias nodes. 
 	*/
	
	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights)
	{
		this.trainingSet=trainingSet;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;
		
		//input layer nodes
		inputNodes=new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		int outputNodeCount=trainingSet.get(0).classValues.size();
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node=new Node(0);
			inputNodes.add(node);
		}
		
		//bias node from input layer to hidden
		Node biasToHidden=new Node(1);
		inputNodes.add(biasToHidden);
		
		//hidden layer nodes
		hiddenNodes=new ArrayList<Node> ();
		for(int i=0;i<hiddenNodeCount;i++)
		{
			Node node=new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int j=0;j<inputNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(inputNodes.get(j),hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}
		
		//bias node from hidden layer to output
		Node biasToOutput=new Node(3);
		hiddenNodes.add(biasToOutput);
			
		//Output node layer
		outputNodes=new ArrayList<Node> ();
		for(int i=0;i<outputNodeCount;i++)
		{
			Node node=new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)
			{
				NodeWeightPair nwp=new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
	}
	
	/**
	 * Get the output from the neural network for a single instance
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2], it should return 1. If outputs
	 * of the outputNodes are [0.1, 0.5, 0.5], it should return 2. 
	 * The parameter is a single instance. 
	 */
	
	public int calculateOutputForInstance(Instance inst)
	{
		// initialize all inputNodes' input values (from instance's attributes)
		for(int i = 0; i < inst.attributes.size(); i++) {
			Node inputNode = this.inputNodes.get(i);
			inputNode.setInput(inst.attributes.get(i));
		}
		
		// calculate outputs at each hidden node
		for(Node hiddenNode : this.hiddenNodes) {
			hiddenNode.calculateOutput();
		}
		
		// calculate outputs at each output node
		for(Node outputNode : this.outputNodes) {
			outputNode.calculateOutput();
		}
		
		// find maximum valued index of outputs
		int indexOfMaxValue = 0; // initialize to index 0
		double valueOfBestIndex = Double.MIN_VALUE; // initialize to negative infinity
		for(int i = 0; i < this.outputNodes.size(); i++) {
			
			// get current node
			Node currentNode = this.outputNodes.get(i);
			
			// check if current node's output is greater or equal to current best
			if(currentNode.getOutput() >= valueOfBestIndex) {
				indexOfMaxValue = i;
				valueOfBestIndex = currentNode.getOutput();
			}
		}
		return indexOfMaxValue;
	}
	
	/**
	 * Train the neural networks with the given parameters
	 * 
	 * The parameters are stored as attributes of this class
	 */
	
	public void train()
	{
		int currentEpoch = 0;
		while( currentEpoch <= this.maxEpoch ) {
			
			for(Instance example : this.trainingSet) {
				
				// propagate forward (calculate all node outputs in network)
				this.calculateOutputForInstance(example);
				
				ArrayList<Double> errorVector = new ArrayList<Double>();
				
				// calculate error vector for each output unit versus expected output at each unit
				for(int i = 0; i < example.classValues.size(); i++) {
					errorVector.add(i, example.classValues.get(i) - this.outputNodes.get(i).getOutput());
				}
				
				// declare 2D array to store hidden to output node ∆W
				double[][] hiddenToOutputWeights = new double[this.hiddenNodes.size()][this.outputNodes.size()];
				
				// compute WJK
				// iterate through output nodes
				for(int k = 0; k < this.outputNodes.size(); k++) {
					Node outputNode = this.outputNodes.get(k);
										
					// iterate through hidden nodes
					for(int j = 0; j < outputNode.parents.size(); j++) {
						NodeWeightPair hiddenNodePair = outputNode.parents.get(j);
						Node hiddenNode = hiddenNodePair.node;
						
						double deltaWJK = this.learningRate * hiddenNode.getOutput() * errorVector.get(k)
								* this.derivativeOfSigmoid(outputNode.getOutput());
						
						hiddenToOutputWeights[j][k] = deltaWJK;
					}
				}
				
				// find sumWJKTKOKGPrimeIn (used for calculating deltaWIJ)
				double[] sumWJKTKOKGPrimeIn = new double[this.hiddenNodes.size()];
				for(int k = 0; k < this.outputNodes.size(); k++) {
					Node outputNode = this.outputNodes.get(k);
					
					for(int j = 0; j < outputNode.parents.size(); j++) {
						NodeWeightPair hiddenNodePair = outputNode.parents.get(j);
						sumWJKTKOKGPrimeIn[j] += hiddenNodePair.weight * errorVector.get(k) * this.derivativeOfSigmoid(outputNode.getOutput());
					}
				}
				
				// declare 2D array to store input to hidden node ∆W
				double[][] inputToHiddenWeights = new double[this.inputNodes.size()][this.hiddenNodes.size()];
				
				// compute deltaWIJ
				for(int j = 0; j < this.hiddenNodes.size() - 1; j++) {
					Node hiddenNode = this.hiddenNodes.get(j);
					
					for(int i = 0; i < hiddenNode.parents.size(); i++) {
						NodeWeightPair inputNodePair = hiddenNode.parents.get(i);
						Node inputNode = inputNodePair.node;
						
						double deltaWIJ = this.learningRate * inputNode.getOutput()
								* this.derivativeOfSigmoid(hiddenNode.getOutput()) * sumWJKTKOKGPrimeIn[j];
												
						inputToHiddenWeights[i][j] = deltaWIJ;
					}
				}
				
				
				// update all the node weights (P,Q part of back-propagation algorithm)...
				
				// first update hidden to output node weights...
				for(int k = 0; k < this.outputNodes.size(); k++) {
					Node outputNode = this.outputNodes.get(k);
					
					for(int j = 0; j < outputNode.parents.size(); j++) {
						NodeWeightPair hiddenNodePair = outputNode.parents.get(j);
						hiddenNodePair.weight += hiddenToOutputWeights[j][k];
					}
				}
				
				// now update input to hidden node weights...
				for(int j = 0; j < this.hiddenNodes.size() - 1; j++) {
					Node hiddenNode = this.hiddenNodes.get(j);
					
					for(int i = 0; i < hiddenNode.parents.size(); i++) {
						NodeWeightPair inputNodePair = hiddenNode.parents.get(i);
						inputNodePair.weight += inputToHiddenWeights[i][j];
					}
				}
			}
			currentEpoch++;
		}
	}
	
	/**
	 * Simple function to return the correct value for the derivative of Sigmoid
	 * based on a certain input.
	 * @param input the input to the derivative of ReLU function.
	 * @return 0 if the input <= 0, or returns 1 if input > 0.
	 */
	public double derivativeOfSigmoid(double input) {
		return (input) * (1 - input);
	}
	
}
