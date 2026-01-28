import React, { useState } from 'react';
import { StyleSheet, View, Text, Image, TouchableOpacity, ScrollView, Alert } from 'react-native';
import { ActivityIndicator } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const API_URL = 'http://192.168.87.233:8000/predict';

export default function Index() {
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // Run prediction on backend
  const runPrediction = async (imageUri: string) => {
    setLoading(true);
    setPrediction(null);

    try {
      // Create form data
      const formData = new FormData();
      
      // @ts-ignore - React Native handles this differently
      formData.append('file', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'photo.jpg',
      });

      // Send to backend
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = await response.json();

      if (response.ok) {
        // Calculate confidence
        const isMalignant = data.probability >= 0.5;
        const confidence = (isMalignant ? data.probability : (1 - data.probability)) * 100;

        // Medical advice
        let advice = '';
        let adviceColor = '';

        if (isMalignant) {
          advice = 'URGENT: This lesion shows characteristics of malignancy. Please consult a dermatologist immediately.';
          adviceColor = '#d32f2f';
        } else if (confidence < 80) {
          advice = 'CAUTION: While this appears benign, confidence is moderate. Recommend professional evaluation.';
          adviceColor = '#f57c00';
        } else {
          advice = 'This lesion appears benign with high confidence. Continue monitoring for changes.';
          adviceColor = '#388e3c';
        }

        setPrediction({
          label: data.label.charAt(0).toUpperCase() + data.label.slice(1),
          confidence: confidence.toFixed(1),
          advice,
          adviceColor
        });
      } else {
        Alert.alert('Error', 'Prediction failed: ' + (data.error || 'Unknown error'));
      }
    } catch (error) {
      console.error('Prediction error:', error);
      Alert.alert('Error', 'Could not connect to server. Make sure your backend is running on the same WiFi network.');
    } finally {
      setLoading(false);
    }
  };

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'We need camera roll permissions!');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      await runPrediction(result.assets[0].uri);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'We need camera permissions!');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      await runPrediction(result.assets[0].uri);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Nevinet</Text>
        <Text style={styles.subtitle}>Skin Lesion Classifier</Text>
      </View>

      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.button} onPress={takePhoto}>
          <Text style={styles.buttonText}>Take Photo</Text>
        </TouchableOpacity>
        
        <TouchableOpacity style={[styles.button, styles.buttonSecondary]} onPress={pickImage}>
          <Text style={styles.buttonText}>Choose from Gallery</Text>
        </TouchableOpacity>
      </View>

      {image && (
        <View style={styles.imageContainer}>
          <Image source={{ uri: image }} style={styles.image} />
        </View>
      )}

      {loading && (
        <View style={styles.loadingPrediction}>
          <ActivityIndicator size="large" color="#1976d2" />
          <Text style={styles.loadingText}>Analyzing image...</Text>
        </View>
      )}

      {prediction && (
        <View style={styles.resultContainer}>
          <View style={styles.resultCard}>
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
          </View>
        </View>
      )}

      <View style={styles.footer}>
        <Text style={styles.footerText}>Only a Prototype</Text>
        <View style={styles.warningBox}>
          <Text style={styles.warningText}>Always consult a healthcare professional</Text>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
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
    textAlign: 'center',
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
  imageContainer: {
    padding: 16,
    paddingTop: 0,
  },
  image: {
    width: '100%',
    height: 300,
    borderRadius: 8,
  },
  loadingPrediction: {
    padding: 32,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  resultContainer: {
    padding: 16,
  },
  resultCard: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 8,
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
    width: '100%',
  },
  footerText: {
    fontSize: 13,
    color: '#666',
    textAlign: 'center',
    marginBottom: 12,
    width: '100%',
    paddingHorizontal: 20,
  },
  warningBox: {
    backgroundColor: '#fee2e2',
    borderWidth: 2,
    borderColor: '#dc2626',
    borderRadius: 6,
    paddingVertical: 10,
    paddingHorizontal: 16,
    width: '90%',
  },
  warningText: {
    fontSize: 13,
    color: '#dc2626',
    fontWeight: '600',
    textAlign: 'center',
  },
});