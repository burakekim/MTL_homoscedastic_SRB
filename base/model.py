import torch
from . import initialization as init

class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_decoder(self.skipless_decoder)
        init.initialize_decoder(self.UnetDecoder_Edge)

        init.initialize_head(self.segmentation_head)
        init.initialize_head(self.edge_head)
        init.initialize_head(self.reconstruct_segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x) # Extract features - common
    ## Segmentation
        decoder_output = self.decoder(*features) # Decoder output for segmentation task
        segmentation_mask = self.segmentation_head(decoder_output) # Feed the decoder output to the segmentation head
    ## Edge
        edge_decoder_output = self.decoder(*features) # Decoder output for edge detection head 
        edge_mask = self.edge_head(edge_decoder_output) # Feed the decoder output to the edge head 
    ## Reconstruction 
        reconstruct_decoder_output = self.skipless_decoder(*features) # Decoder output for reconstruction task
        reconstruction_mask = self.reconstruct_segmentation_head(reconstruct_decoder_output) # Feed the decoder output to the reconstruction head

        return segmentation_mask, edge_mask, reconstruction_mask, self.sigma

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            segmentation_mask, edge_mask, reconstruction_mask, self.sigma  = self.forward(x)

        return segmentation_mask, edge_mask, reconstruction_mask