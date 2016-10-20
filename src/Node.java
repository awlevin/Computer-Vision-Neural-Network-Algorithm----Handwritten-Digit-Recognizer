/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details
 * 
 * Do not modify. 
 */


import java.util.*;

public class Node{
	private int type=0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
	public ArrayList<NodeWeightPair> parents=null; //Array List that will contain the parents (including the bias node) with weights if applicable
		 
	private Double inputValue=0.0;
	private Double outputValue=0.0;
	private Double sum=0.0; // sum of wi*xi
	
	//Create a node with a specific type
	public Node(int type)
	{
		if(type>4 || type<0)
		{
			System.out.println("Incorrect value for node type");
			System.exit(1);
			
		}
		else
		{
			this.type=type;
		}
		
		if (type==2 || type==4)
		{
			parents=new ArrayList<NodeWeightPair>();
		}
	}
	
	//For an input node sets the input value which will be the value of a particular attribute
	public void setInput(Double inputValue)
	{
		if(type==0)//If input node
		{
			this.inputValue=inputValue;
		}
	}
	
	/**
	 * Calculate the output of a ReLU node.
	 * You can assume that outputs of the parent nodes have already been calculated
	 * You can get this value by using getOutput()
	 * @param train: the training set
	 */
	public void calculateOutput()
	{
		
		if(type==2 || type==4)//Not an input or bias node
		{
			sum = 0.0;
			NodeWeightPair parentNodeWeightPair = null;
			
			// Calculate the sum of (inputValues * weights)
			for(int i = 0; i < this.parents.size(); i++) {
				parentNodeWeightPair = this.parents.get(i);
				double parentWeight = parentNodeWeightPair.weight;
				double outputOfParent = parentNodeWeightPair.node.getOutput();
				
//				System.out.println("parentWeight: " + parentWeight + "\tparentOutput: " + outputOfParent); 
				
				this.sum += outputOfParent * parentWeight;
			}
			
			// set outputValue equal to g(x) activation function (here we use Sigmoid)
			this.outputValue = 1/(1 + Math.exp(-sum));
		}
	}

	public double getSum() {
		return sum;
	}
	
	//Gets the output value
	public double getOutput()
	{
		
		if(type==0)//Input node
		{
			return inputValue;
		}
		else if(type==1 || type==3)//Bias node
		{
			return 1.00;
		}
		else
		{
			return outputValue;
		}
		
	}
}


