from django import forms
from .models import PatientRecord


class PatientRecordForm(forms.ModelForm):
    class Meta:
        model = PatientRecord
        fields = '__all__'  # Ajusta según necesites


class MultipleRecordsForm(forms.Form):
    file = forms.FileField(label='Subir archivo CSV')