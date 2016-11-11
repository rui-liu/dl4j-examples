package org.deeplearning4j.examples.cbu;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.convolution.Convolution;

/**
 * Created by prt on 16-11-9.
 */
public class ToutiaoExample {
    public void main(String[] args) throws Exception{
        new ToutiaoExample().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .learningRate(0.01)
            .graphBuilder()
            .addInputs("input").addLayer("L1", new EmbeddingLayer.Builder().nIn(50000).nOut(50).build())
            .inputPreProcessor("L1", new FeedForwardToCnnPreProcessor(1000, 50))
            .addLayer("L2_2", new ConvolutionLayer.Builder(2,50).nOut(16).build(), "L1")
            .addLayer("L3_2", new SubsamplingLayer.Builder().kernelSize(999,1).build(), "L2_2")
            .addLayer("L2_3", new ConvolutionLayer.Builder(3,50).nOut(16).build(), "L1")
            .addLayer("L3_3", new SubsamplingLayer.Builder().kernelSize(998,1).build(), "L2_3")
            .addLayer("L2_4", new ConvolutionLayer.Builder(4,50).nOut(16).build(), "L1")
            .addLayer("L3_4", new SubsamplingLayer.Builder().kernelSize(997,1).build(), "L2_4")
            .addVertex("L4", new MergeVertex(), "L3_2","L3_3", "L3_4")
            .addLayer("Output", new OutputLayer.Builder().nIn(48).nOut(27).build(), "L4").backprop(true).pretrain(false).build();
        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
    }

}
