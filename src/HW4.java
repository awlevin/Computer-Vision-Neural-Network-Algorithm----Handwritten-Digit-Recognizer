import java.util.*;
import java.io.BufferedReader;
import java.io.FileReader;

/**
 * Do not modify.
 * 
 * This is the class with the main function
 */

public class HW4{
	/**
	 * Runs the tests for HW4
	*/
	public static void main(String[] args)
	{
		//Checking for correct number of arguments
		if (args.length < 5) 
		{
			System.out.println("usage: java HW4 <noHiddenNode> " +
					"<learningRate> <maxEpoch> <trainFile> <testFile>");
			System.exit(-1);
		}
		
		//Reading the training set 	
		ArrayList<Instance> trainingSet=getData(args[3]);
		
		
		//Reading the weights
		Double[][] hiddenWeights=new Double[Integer.parseInt(args[0])][];
		
		for(int i=0;i<hiddenWeights.length;i++)
		{
			hiddenWeights[i]=new Double[trainingSet.get(0).attributes.size()+1];
		}
		
		Double [][] outputWeights=new Double[trainingSet.get(0).classValues.size()][];
		for (int i=0; i<outputWeights.length; i++) {
			outputWeights[i]=new Double[hiddenWeights.length+1];
		}
		
		readWeights(hiddenWeights,outputWeights);
		
		Double learningRate=Double.parseDouble(args[1]);
		
		if(learningRate>1 || learningRate<=0)
		{
			System.out.println("Incorrect value for learning rate\n");
			System.exit(-1);
		}
		
		NNImpl nn=new NNImpl(trainingSet,Integer.parseInt(args[0]),Double.parseDouble(args[1]),Integer.parseInt(args[2]), 
					hiddenWeights,outputWeights);
		nn.train();
			
		//Reading the training set 	
		ArrayList<Instance> testSet=getData(args[4]);
			
		Integer[] outputs=new Integer[testSet.size()];
			
			
		int correct=0;
		for(int i=0;i<testSet.size();i++)
		{
			//Getting output from network
			outputs[i]=nn.calculateOutputForInstance(testSet.get(i));
			int actual_idx=-1;
			for (int j=0; j < testSet.get(i).classValues.size(); j++) {
				if (testSet.get(i).classValues.get(j) > 0.5)
					actual_idx=j;
			}
				
			if(outputs[i] == actual_idx)
			{
				correct++;
			} else {
				System.out.println(i + "th instance got an misclassification, expected: " + actual_idx + ". But actual:" + outputs[i]);
			}
		}
			
			System.out.println("Total instances: " + testSet.size());
			System.out.println("Correctly classified: "+correct);
			
	}
		
	// Reads a file and gets the list of instances
	private static ArrayList<Instance> getData(String file)
	{
		ArrayList<Instance> data=new ArrayList<Instance>();
		BufferedReader in;
		int attributeCount=0;
		int outputCount=0;
		
		try{
			in = new BufferedReader(new FileReader(file));
			while (in.ready()) { 
				String line = in.readLine(); 	
				String prefix = line.substring(0, 2);
				if (prefix.equals("//")) {
				} 
				else if (prefix.equals("##")) {
					attributeCount=Integer.parseInt(line.substring(2));
				} else if (prefix.equals("**")) {
					outputCount=Integer.parseInt(line.substring(2));
				}else {
					String[] vals=line.split(" ");
					Instance inst = new Instance();
					for (int i=0; i<attributeCount; i++)
						inst.attributes.add(Double.parseDouble(vals[i]));
					for (int i=attributeCount; i < vals.length; i++)
						inst.classValues.add(Integer.parseInt(vals[i]));	
					data.add(inst);	
				}
				
			}
			in.close();
			return data;
			
		}catch(Exception e)
		{
			System.out.println("Could not read instances: "+e);
		}
		
		return null;
	}
	// Gets weights randomly
	public static void readWeights(Double [][]hiddenWeights, Double[][]outputWeights)
	{
		Random r = new Random();
			
		for(int i=0;i<hiddenWeights.length;i++)
		{
			for(int j=0;j<hiddenWeights[i].length;j++)
			{
				hiddenWeights[i][j] = r.nextDouble()*0.01;
			}
		}
				
		for(int i=0;i<outputWeights.length;i++)
		{
			for (int j=0; j<outputWeights[i].length; j++)
			{
				outputWeights[i][j] = r.nextDouble()*0.01;
			}
		}	
	

	}
}
