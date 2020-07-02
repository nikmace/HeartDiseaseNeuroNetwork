using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworks;
using System.IO;

namespace NeuralNetworksTest
{
    [TestClass()]
    public class NeuralNetworksTest
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var outputs = new double[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            var inputs = new double[,]
            {
                // Result - Patient is Healthy - 1
                //          Patient is Sick - 0
                //
                // Has Temperature T
                // Good Age
                // Smokes S
                // Eats Healthy - F

               //T A S F
                {0,0,0,0},
                {0,0,0,1},
                {0,0,1,0},
                {0,0,1,1},
                {0,0,0,0},
                {0,1,0,0},
                {0,1,0,1},
                {0,1,1,1},
                {1,0,0,0},
                {1,0,0,1},
                {1,0,1,0},
                {1,0,1,1},
                {1,1,0,0},
                {1,1,0,1},
                {1,1,1,0},
                {1,1,1,1}

            };
            //               Input Layer|
            //                  Ouput Layer|
            //                    Learning Rate|
            //                         Hidden Layer|
            var topology = new Topology(4, 1, 0.1, 2);
            var neuralNetwork = new NeuralNetwork(topology);

            var difference = neuralNetwork.Learn(outputs, inputs, 10000); //Neural network was learning here

            var results = new List<double>();
            for(int i = 0; i < outputs.Length; i++)
            {
                var row = NeuralNetwork.GetRow(inputs, i);
                var res = neuralNetwork.FeedForward(row).Output;
                // Here we're using th NeuralNetwork and getting results
                results.Add(res);
                
            }
            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 4);
                var actual = Math.Round(results[i], 4);
                Assert.AreEqual(expected, actual);
            }

        }

        [TestMethod()]
        public void DataSetTest()
        {
            var outputs = new List<double>();
            var inputs = new List<double[]>();
            using (var sr = new StreamReader("heart.csv"))
            {
                var header = sr.ReadLine();

                while (!sr.EndOfStream)
                {
                    var row = sr.ReadLine();
                    var values = row.Split(',').Select(v => Convert.ToDouble(v.Replace(".",","))).ToList();
                    var output = values.Last();
                    var input = values.Take(values.Count - 1).ToArray();

                    outputs.Add(output);
                    inputs.Add(input);
                }
            }

            var inputSignals = new double[inputs.Count, inputs[0].Length];
            for (int i = 0; i < inputSignals.GetLength(0); i++)
            {
                for (var j = 0; j < inputSignals.GetLength(1); j++)
                {
                    inputSignals[i, j] = inputs[i][j];
                }
            }
            var topology = new Topology(outputs.Count, 1, 0.1, outputs.Count / 2);
            var neuralNetwork = new NeuralNetwork(topology);

            var difference = neuralNetwork.Learn(outputs.ToArray(), inputSignals, 100);

            var results = new List<double>();
            for (int i = 0; i < outputs.Count; i++)
            {
                var res = neuralNetwork.FeedForward(inputs[i]).Output;
                // Here we're using the NeuralNetwork and getting results
                results.Add(res);

            }
            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 2);
                var actual = Math.Round(results[i], 2);
                Assert.AreEqual(expected, actual);
            }
        }
    }
}
