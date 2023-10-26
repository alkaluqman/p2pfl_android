package com.android.example.nnapi.gettingweights;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.os.FileUtils;
import android.provider.OpenableColumns;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.support.metadata.schema.TensorMetadata;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.gpu.CompatibilityList;

public class MainActivity extends AppCompatActivity {
    String filePath;
    TextView status, inputs, outputs;
    Button selectFile;
    private Interpreter interpreter;
    private static final int FILE_REQUEST_CODE = 128;

    protected void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        selectFile = findViewById(R.id.selectFile);
        status = findViewById(R.id.status_title);
        inputs = findViewById(R.id.input);
        outputs = findViewById(R.id.output);

        selectFile.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                //intent.addCategory(Intent.CATEGORY_OPENABLE);
                //intent.setType("*/*");
                //startActivityForResult(intent,FILE_REQUEST_CODE);
                try {
                    interpreter = new Interpreter(loadModelFile("mobilenetv1.tflite"));
                    MetadataExtractor metadataExtractor = new MetadataExtractor(loadModelFile("mobilenetv1.tflite"));
                    //ByteBuffer inputBuffer = metadataExtractor.getInputTensorMetadata(0).nameAsByteBuffer();
                    if(metadataExtractor.hasMetadata()){
                        System.out.println("Yes");
                        System.out.println("Input Tensor Count: " + metadataExtractor.getInputTensorCount());
                        System.out.println("Output Tensor Count: " + metadataExtractor.getOutputTensorCount());

                        System.out.println("Input Tensor Shape:" + Arrays.toString(metadataExtractor.getInputTensorShape(0)));
                        System.out.println("Input Tensor TensorMetadata name:" + metadataExtractor.getInputTensorMetadata(0).name());
                        System.out.println("Input Tensor Quantization Params Scale: " + metadataExtractor.getInputTensorQuantizationParams(0).getScale());
                        System.out.println("Input Tensor Quantization Params Zero Point: " + metadataExtractor.getInputTensorQuantizationParams(0).getZeroPoint());

                        for(int i=0;i!=metadataExtractor.getOutputTensorCount();i++){
                            System.out.println("Output Tensor Shape:" + Arrays.toString(metadataExtractor.getOutputTensorShape(i)));
                            System.out.println("Output Tensor TensorMetadata name:" + metadataExtractor.getOutputTensorMetadata(i).name());
                            System.out.println("Output Tensor Quantization Params Scale: " + metadataExtractor.getOutputTensorQuantizationParams(i).getScale());
                            System.out.println("Output Tensor Quantization Params Zero Point: " + metadataExtractor.getOutputTensorQuantizationParams(i).getZeroPoint());
                        }
                    } else {
                        System.out.println("No");
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }

                // running on Python script on Java
                /*try {

                    String pythonScript = "assets/model.py";

                    ProcessBuilder processBuilder = new ProcessBuilder("python", pythonScript);

                    Process process = processBuilder.start();

                    int exitCode = process.waitFor();
                    System.out.println("Python script exited with code:" + exitCode);

                    Interpreter interpreter = new Interpreter(loadModelFile("model.tflite"));
                    int inputTensorIndex = 0;
                    int[] inputShape = interpreter.getInputTensor(inputTensorIndex).shape();
                    DataType inputDataType = interpreter.getInputTensor(inputTensorIndex).dataType();

                    System.out.println("Input Shape: "+ Arrays.toString(inputShape));
                    System.out.println("Input Data Type: "+ inputDataType);

                    int outputTensorIndex = 0;
                    int[] outputShape = interpreter.getOutputTensor(outputTensorIndex).shape();
                    DataType outputDataType = interpreter.getOutputTensor(outputTensorIndex).dataType();

                    System.out.println("Output Shape: "+ Arrays.toString(outputShape));
                    System.out.println("Output Data Type: "+ outputDataType);

                    interpreter.close();
                } catch (IOException | InterruptedException e) {
                    e.printStackTrace();
                }*/
            }
        });
    }

    private void saveCheckpoint(){
        // Conduct training jobs

        // Export the trained weights as a checkpoint file
        File outputFile = new File(getFilesDir(), "checkpoint.ckpt");
        Map<String, Object> inputs = new HashMap<>();
        inputs.put("checkpoint_path", outputFile.getAbsolutePath());
        Map<String, Object> outputs = new HashMap<>();
        interpreter.runSignature(inputs, outputs, "save");
    }

    private void restoreCheckpoint() throws IOException {
        try (Interpreter anotherInterpreter = new Interpreter(loadModelFile("model.tflite"))) {
            // Load the trained weights from the checkpoint file.
            File outputFile = new File(getFilesDir(), "checkpoint.ckpt");
            Map<String, Object> inputs = new HashMap<>();
            inputs.put("checkpoint_path", outputFile.getAbsolutePath());
            Map<String, Object> outputs = new HashMap<>();
            anotherInterpreter.runSignature(inputs, outputs, "restore");
        }
    }

    private void getWeights(){
        Model.Options options;
        CompatibilityList compatList = new CompatibilityList();

        if(compatList.isDelegateSupportedOnThisDevice()){
            // if the device has a supported GPU, add the GPU delegate
            options = new Model.Options.Builder().setDevice(Model.Device.GPU).build();
        } else {
            // if the GPU is not supported, run on 4 threads
            options = new Model.Options.Builder().setNumThreads(4).build();
        }

        int numInputs = interpreter.getInputTensorCount();
        String[] key = interpreter.getSignatureKeys();
        //Tensor numInputs1 = interpreter.getInputTensor(0);
        int numOutputs = interpreter.getOutputTensorCount();
        //Tensor numOutputs1 = interpreter.getOutputTensor(1);
        System.out.println("keys: "+key);

        inputs.setText("Number of inputs: " + numInputs);
        outputs.setText("Number of outputs: " + numOutputs);

        for (int i = 0; i < numInputs; i++) {
            Tensor inputTensor = interpreter.getInputTensor(i);
            //ByteBuffer byteBuffer = inputTensor.getBuffer();
            int[] shape = inputTensor.shape();
            System.out.println("Input shape: " + Arrays.toString(shape));
            System.out.println("Input tensor: " + inputTensor);
            //inputs.setText("Input tensor shape: " + Arrays.toString(shape));
            //inputs.setText("Get first few bytes of the input tensor: " + Arrays.toString(getFirstFewBytes(byteBuffer)));
        }

        for (int i = 0; i < numOutputs; i++) {
            Tensor outputTensor = interpreter.getOutputTensor(i);
            int[] shape = outputTensor.shape();
            System.out.println("Output shape: " +Arrays.toString(shape));
            System.out.println("Output tensor: " + outputTensor);
            //outputs.setText("Output tensor shape: " + Arrays.toString(shape));
            //ByteBuffer byteBuffer = outputTensor.getBuffer();
        }

    }

    private MappedByteBuffer loadModelFile(String modelPath) throws IOException {
        AssetManager assetManager = getAssets();
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private String getFilePathFromUri(Uri uri){
        String filePath = null;
        String[] projection = {OpenableColumns.DISPLAY_NAME};
        try{
            Cursor cursor = getContentResolver().query(uri,projection,null,null,null);
            if(cursor!=null){
                int column_index = cursor.getColumnIndexOrThrow(OpenableColumns.DISPLAY_NAME);
                cursor.moveToFirst();
                filePath = cursor.getString(column_index);
                cursor.close();
            }
        } catch (Exception e){
            e.printStackTrace();
        }
        return filePath;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data){
        super.onActivityResult(requestCode,resultCode,data);

        if(requestCode == FILE_REQUEST_CODE && resultCode == RESULT_OK && data!=null){
            if(data.getData() != null){
                Uri fileUri = data.getData();
                filePath = getFilePathFromUri(fileUri);
                status.setText("File Path: "+filePath);
            }
            /*
            filepath = new File(data.getData().getPath()).getAbsolutePath();
            Uri selectedFileUri = data.getData();

            String filename;
            Cursor cursor = getContentResolver().query(selectedFileUri,null,null,null,null);
            if(cursor == null){
                filename = selectedFileUri.getPath();
            } else {
                cursor.moveToFirst();
                int idx = cursor.getColumnIndex(MediaStore.Files.FileColumns.DISPLAY_NAME);
                filename = cursor.getString(idx);
                cursor.close();
            }
            fileName = filename.substring(0,filename.lastIndexOf("."));
            extension = filename.substring(filename.lastIndexOf(".")+1);
            status.setText("Selected file: "+fileName+"."+extension);*/
        }
    }
}
