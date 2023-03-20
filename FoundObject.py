class FoundObject:
  def __init__(self, name, confidence):
    self.name = name
    self.confidence = confidence
  def __str__(self):
    return self.name + " " + str(self.confidence)