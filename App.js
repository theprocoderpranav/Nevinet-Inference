// App.js - Nevinet Mobile App
import React, { useState, useEffect } from 'react';
import { StyleSheet, View, Text, Image, TouchableOpacity, ScrollView, Platform } from 'react-native';
import { Button, Card, ActivityIndicator } from 'react-native-paper';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';

export default function App() {
  const [session, setSession] = useState(null);
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelLoading, setModelLoading] = useState(true);

  // Load ONNX model
  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    try {
      console.log('Loading ONNX model...');
      
      // Copy model to cache directory (ONNX Runtime needs file path)
      const modelAsset = require('./assets/model/nevinet_mobile.onnx');
      const modelPath = `${FileSystem.cacheDirectory}nevinet_mobile.onnx`;
      
      // Copy from assets to cache
      await FileSystem.downloadAsync(
        modelAsset,
        modelPath
      );
      
      // Create inference session
      const newSession = await InferenceSession.create(modelPath);
      setSession(newSession);
      setModelLoading(false);
      console.log('✅ Model loaded successfully!');
    } catch (error) {
      console.error('Error loading model:', error);
      alert('Failed to load model: ' + error.message);
      setModelLoading(false);
    }
  };

  // Image preprocessing
  const preprocessImage = async (uri) => {
    // Load image
    const response = await fetch(uri);
    const blob = await response.blob();
    const img = await createImageBitmap(blob);
    
    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    
    // Draw and resize image
    ctx.drawImage(img, 0, 0, 224, 224);
    
    // Get pixel data
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const pixels = imageData.data;
    
    // Normalize to [-1, 1] (ImageNet normalization)
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    const float32Data = new Float32Array(3 * 224 * 224);
    
    for (let i = 0; i < 224 * 224; i++) {
      // RGB channels
      float32Data[i] = ((pixels[i * 4] / 255) - mean[0]) / std[0];           // R
      float32Data[224 * 224 + i] = ((pixels[i * 4 + 1] / 255) - mean[1]) / std[1];  // G
      float32Data[2 * 224 * 224 + i] = ((pixels[i * 4 + 2] / 255) - mean[2]) / std[2];  // B
    }
    
    return float32Data;
  };

  // Run inference
  const runInference = async (imageUri) => {
    if (!session) {
      alert('Model not loaded yet!');
      return;
    }

    setLoading(true);
    setPrediction(null);

    try {
      // Preprocess image
      const inputData = await preprocessImage(imageUri);
      
      // Create input tensor
      const inputTensor = new Tensor('float32', inputData, [1, 3, 224, 224]);
      
      // Run inference
      const feeds = { input: inputTensor };
      const results = await session.run(feeds);
      
      // Get output (logit)
      const output = results.output.data[0];
      
      // Apply sigmoid to get probability
      const probability = 1 / (1 + Math.exp(-output));
      
      // Classify
      const isMalignant = probability >= 0.5;
      const label = isMalignant ? 'Malignant' : 'Benign';
      const confidence = (isMalignant ? probability : (1 - probability)) * 100;
      
      // Medical advice
      let advice = '';
      let adviceColor = '';
      
      if (isMalignant) {
        advice = '⚠️ URGENT: This lesion shows characteristics of malignancy. Please consult a dermatologist immediately.';
        adviceColor = '#d32f2f';
      } else if (confidence < 80) {
        advice = '⚠️ CAUTION: While this appears benign, confidence is moderate. Recommend professional evaluation.';
        adviceColor = '#f57c00';
      } else {
        advice = '✓ This lesion appears benign with high confidence. Continue monitoring for changes.';
        adviceColor = '#388e3c';
      }
      
      setPrediction({
        label,
        probability,
        confidence: confidence.toFixed(1),
        advice,
        adviceColor
      });
      
    } catch (error) {
      console.error('Inference error:', error);
      alert('Prediction failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Pick image from gallery
  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (status !== 'granted') {
      alert('Sorry, we need camera roll permissions!');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      await runInference(result.assets[0].uri);
    }
  };

  // Take photo with camera
  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    
    if (status !== 'granted') {
      alert('Sorry, we need camera permissions!');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      await runInference(result.assets[0].uri);
    }
  };

  if (modelLoading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#1976d2" />
        <Text style={styles.loadingText}>Loading Nevinet model...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Nevinet</Text>
        <Text style={styles.subtitle}>Skin Lesion Classifier</Text>
      </View>

      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.button} onPress={takePhoto}>
          <Text style={styles.buttonText}>📷 Take Photo</Text>
        </TouchableOpacity>
        
        <TouchableOpacity style={[styles.button, styles.buttonSecondary]} onPress={pickImage}>
          <Text style={styles.buttonText}>🖼️ Choose from Gallery</Text>
        </TouchableOpacity>
      </View>

      {image && (
        <Card style={styles.card}>
          <Card.Cover source={{ uri: image }} />
        </Card>
      )}

      {loading && (
        <View style={styles.loadingPrediction}>
          <ActivityIndicator size="large" color="#1976d2" />
          <Text style={styles.loadingText}>Analyzing image...</Text>
        </View>
      )}

      {prediction && (
        <View style={styles.resultContainer}>
          <Card style={styles.resultCard}>
            <Card.Content>
              <Text style={styles.resultLabel}>Prediction:</Text>
              <Text style={[
                styles.resultValue,
                { color: prediction.label === 'Malignant' ? '#d32f2f' : '#388e3c' }
              ]}>
                {prediction.label}
              </Text>
              
              <Text style={styles.resultLabel}>Confidence:</Text>
              <Text style={styles.resultValue}>{prediction.confidence}%</Text>
              
              <View style={[styles.adviceBox, { backgroundColor: prediction.adviceColor + '20' }]}>
                <Text style={[styles.adviceText, { color: prediction.adviceColor }]}>
                  {prediction.advice}
                </Text>
              </View>
            </Card.Content>
          </Card>
        </View>
      )}

      <View style={styles.footer}>
        <Text style={styles.footerText}>⚕️ For educational purposes only</Text>
        <Text style={styles.footerText}>Always consult a healthcare professional</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  header: {
    backgroundColor: '#1976d2',
    padding: 24,
    paddingTop: 60,
    alignItems: 'center',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: 'white',
  },
  subtitle: {
    fontSize: 16,
    color: 'white',
    opacity: 0.9,
    marginTop: 4,
  },
  buttonContainer: {
    padding: 16,
    gap: 12,
  },
  button: {
    backgroundColor: '#1976d2',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonSecondary: {
    backgroundColor: '#424242',
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  card: {
    margin: 16,
    marginTop: 0,
  },
  loadingPrediction: {
    padding: 32,
    alignItems: 'center',
  },
  resultContainer: {
    padding: 16,
  },
  resultCard: {
    backgroundColor: 'white',
  },
  resultLabel: {
    fontSize: 14,
    color: '#666',
    marginTop: 12,
  },
  resultValue: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 4,
  },
  adviceBox: {
    marginTop: 16,
    padding: 16,
    borderRadius: 8,
  },
  adviceText: {
    fontSize: 14,
    lineHeight: 20,
  },
  footer: {
    padding: 24,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 12,
    color: '#999',
    textAlign: 'center',
  },
});