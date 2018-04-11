import java.io.*;
import java.util.*;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;
	public class mymapper extends MapReduceBase implements Mapper<NullWritable, Text, IntWritable, DoubleWritable>	{

			private JobConf conf;

			@Override
			public void configure(JobConf conf)
			{
				this.conf = conf;

			}
			public void map(NullWritable key, Text value, OutputCollector<IntWritable, DoubleWritable> output, Reporter reporter) throws IOException
			{
				FSDataInputStream currentStream;
    			BufferedReader currentReader;
				FileSystem fs;
				Path path = new Path(value.toString()); // get file path
				fs = path.getFileSystem(conf);			// initiate filesystem
				currentStream = fs.open(path);			// Open FSDataInputStream
      			currentReader = new BufferedReader(new InputStreamReader(currentStream)); // Get bufferreader to start reading

				// From here write your code to read data from input file using BufferedReader of Java. Construct
				// Graph while reading and count triangle and open triple from the graph to compute
				// Clustering coefficient. At the end, prepare your output (key,value) pair to pass to reducer.


				
				IntWritable key1 = new IntWritable(1);
				DoubleWritable value1 = new DoubleWritable(2.5);
				output.collect(key1, value1); 
				
			}

		}
