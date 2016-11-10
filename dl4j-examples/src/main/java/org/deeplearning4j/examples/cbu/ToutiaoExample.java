package org.deeplearning4j.examples.cbu;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
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
            .addLayer("L2_2", new ConvolutionLayer.Builder(2,50).nOut(16).build(), "L1")
            .addLayer("L3_2", new SubsamplingLayer())
            .addLayer("L2_3", new ConvolutionLayer.Builder(3,50).nOut(16).build(), "L1")
            .addLayer("L2_4", new ConvolutionLayer.Builder(4,50).nOut(16).build(), "L1")
            .addVertex()
    }

    Layer poolConv() {
        Layer c = new ConvolutionLayer.Builder(2,50).nIn(1).nOut(16).build();
    }
}
