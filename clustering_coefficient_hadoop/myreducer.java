import java.io.*;
import java.util.*;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;
//Reducer to merge all types of graphlet count
		public class myreducer extends MapReduceBase implements Reducer<IntWritable, DoubleWritable, Text, DoubleWritable> {

			public void reduce(IntWritable key, Iterator<DoubleWritable> values, OutputCollector<Text, DoubleWritable> output, Reporter arg3)
					throws IOException {

				
				Text output_key  = new Text("Test");
				DoubleWritable output_value = new DoubleWritable(2.5);
				output.collect(output_key, output_value);
				
						
				}
			}
