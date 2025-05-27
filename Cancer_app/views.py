from django.shortcuts import render, redirect
from .forms import PatientRecordForm, MultipleRecordsForm
from .models import PatientRecord
from .ml_model import BreastCancerModel
import pandas as pd
from django.contrib import messages

ml_model = BreastCancerModel()


def home(request):
    return render(request, 'home.html')


def single_prediction(request):
    if request.method == 'POST':
        form = PatientRecordForm(request.POST)
        if form.is_valid():
            record = form.save(commit=False)

            # Convertir datos a formato para el modelo
            input_data = pd.DataFrame([[
                record.mean_radius,
                record.mean_texture,
                # Agrega todos los campos necesarios
            ]], columns=[  # Asegúrate que los nombres coincidan con los que espera el modelo
                'mean radius',
                'mean texture',
                # ...
            ])

            prediction, probabilities = ml_model.predict(input_data)

            record.prediction = prediction[0]
            record.malignant_prob = probabilities[0][1]  # Probabilidad de maligno
            record.benign_prob = probabilities[0][0]  # Probabilidad de benigno
            record.save()

            return render(request, 'result.html', {
                'record': record,
                'single': True
            })
    else:
        form = PatientRecordForm()

    return render(request, 'single_prediction.html', {'form': form})


def multiple_predictions(request):
    if request.method == 'POST':
        form = MultipleRecordsForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                df = pd.read_csv(request.FILES['file'])
                # Asegúrate que el CSV tiene las columnas correctas

                predictions, probabilities = ml_model.predict(df)

                records = []
                for i, row in df.iterrows():
                    record = PatientRecord(
                        mean_radius=row['mean_radius'],
                        mean_texture=row['mean_texture'],
                        # ...
                        prediction=predictions[i],
                        malignant_prob=probabilities[i][1],
                        benign_prob=probabilities[i][0]
                    )
                    record.save()
                    records.append(record)

                return render(request, 'result.html', {
                    'records': records,
                    'single': False
                })
            except Exception as e:
                messages.error(request, f"Error procesando el archivo: {str(e)}")
    else:
        form = MultipleRecordsForm()

    return render(request, 'multiple_predictions.html', {'form': form})