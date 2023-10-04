package com.android.example.nnapi.filetransfer;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import com.android.example.nnapi.filetransfer.Communications.BluetoothActivity;
import com.android.example.nnapi.filetransfer.Communications.WifiActivity;

public class MainActivity extends AppCompatActivity {

    Button wifi,bluetooth;
    protected void onCreate(Bundle savedInstanceState){
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        wifi = findViewById(R.id.wifi);
        bluetooth = findViewById(R.id.bluetooth);

        wifi.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openWifiActivity();
            }
        });

        bluetooth.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openBluetoothActivity();
            }
        });
    }

    public void openBluetoothActivity(){
        Intent intent = new Intent(this, BluetoothActivity.class);
        startActivity(intent);
    }

    public void openWifiActivity() {
        Intent intent = new Intent(this, WifiActivity.class);
        startActivity(intent);
    }
}
