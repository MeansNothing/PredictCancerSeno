from django.db import models


class PatientRecord(models.Model):
    # Campos básicos (ajusta según las características de tu modelo)
    mean_radius = models.FloatField()
    mean_texture = models.FloatField()
    mean_perimeter = models.FloatField()
    mean_area = models.FloatField()
    mean_smoothness = models.FloatField()
    # Agrega todos los campos necesarios según tu dataset

    prediction = models.IntegerField(null=True, blank=True)
    malignant_prob = models.FloatField(null=True, blank=True)
    benign_prob = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Record {self.id} - {'Malignant' if self.prediction else 'Benign'}"