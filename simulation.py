"""_summary_
Bu kod, bir ad hoc ağ simülasyonu yürütmek için kullanılabilir. Ad hoc ağlar, belirli bir yönlendirme altyapısı olmaksızın birbirine bağlı bir dizi düğümden oluşan kablosuz ağlardır. Bu kodda, belirli bir hareket modeline göre düğümlerin hareket ettiği ve belirli bir iletim menziline sahip oldukları varsayılır. Ayrıca, belirli bir zamanda düğümler arasında mesaj gönderimini taklit ederek, ağda iletilen mesajların oranını hesaplar ve görselleştirir.
Kod, AdHocNode ve AdHocNetwork adlı iki sınıf içerir. AdHocNode sınıfı, bir düğümün benzersiz kimliği (node_id) ve koordinatları (x ve y) ile temsil edilir. AdHocNetwork sınıfı, AdHocNode örneklerinden oluşan bir düğüm listesiyle birlikte gelir. Ayrıca, belirli bir hareket modeline (move_model_type) ve belirli bir iletim menziline (transmission_range) sahiptir. Bir simülasyon çalıştırılacağı zaman, düğümlerin hareket modeli kullanılarak rastgele hareket ettirilir ve ardından düğümler arasındaki mesajlaşma simüle edilir.
Kod, ayrıca farklı hareket modellerini temsil etmek için HareketModeli adlı bir soyut sınıf içerir. Bu soyut sınıftan türetilen alt sınıflar, farklı hareket modellerini uygular. Örneğin, RastgeleHareket sınıfı, düğümleri belirli bir maksimum adımda rastgele hareket ettirirken, GaussMarkov sınıfı, önceki hareketin bir ölçüsüne dayanarak mevcut hareketi belirler.
Kod, ayrıca simülasyon sonuçlarının görselleştirilmesini sağlamak için matplotlib kütüphanesini kullanır.
    """


import random
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import imageio
from matplotlib.animation import PillowWriter

class AdHocNode:
    def __init__(self, node_id, x, y):
        self.node_id = node_id
        self.x = x
        self.y = y

    def move(self, dx, dy):
        self.x += dx
        self.y += dy


class AdHocNetwork:
    def __init__(self, num_nodes, move_model_type):
        self.nodes = [AdHocNode(i, random.uniform(0, 100), random.uniform(0, 100)) for i in range(num_nodes)]
        self.move_model_type = move_model_type
        self.move_model = self._get_move_model()
        self.delivered_msgs = 0
        self.total_msgs = 0
        
    def _get_move_model(self):
        if self.move_model_type == "rastgele":
            return RastgeleHareket(transmission_range=10, max_step_size=5)
        elif self.move_model_type == "gaus-markov":
            return GaussMarkov(transmission_range=10, alpha=0.5,beta=0.25)
        elif self.move_model_type == "manzara":
            return Manzara(transmission_range=10, num_landmarks=5,landmark_range=5)
        elif self.move_model_type == "ruzgar":
            return Ruzgar(transmission_range=10,wind_speed=10,wind_dir=45,max_step_size=5)
        elif self.move_model_type == "izle-ve-hareket-et":
            return IzleVeHareketEt(transmission_range=10, max_speed=5)
        elif self.move_model_type == "Sismik":
            return Sismik(transmission_range=10,magnitude=5)
        else:
            raise ValueError("Geçersiz hareket modeli tipi")
            
    def send_message(self, src_node_id, dest_node_id):
        self.total_msgs += 1
        src_node = self.nodes[src_node_id]
        dest_node = self.nodes[dest_node_id]
        dist = ((src_node.x - dest_node.x)**2 + (src_node.y - dest_node.y)**2)**0.5
        if dist <= self.move_model.transmission_range:
            self.delivered_msgs += 1
        
    def run_simulation(self, num_iterations, show_visualization=True):
        for i in range(num_iterations):
            for node in self.nodes:
                dx, dy = self.move_model.get_displacement(node.x, node.y)
                node.move(dx, dy)
            for src_node_id in range(len(self.nodes)):
                for dest_node_id in range(len(self.nodes)):
                    if src_node_id != dest_node_id:
                        self.send_message(src_node_id, dest_node_id)
            if show_visualization:
                self.visualize_network()
    
    def visualize_network(self):
        fig, ax = plt.subplots()
        x = [node.x for node in self.nodes]
        y = [node.y for node in self.nodes]
        scat = ax.scatter(x, y)

        def update(frame_number):
            for i in range(len(self.nodes)):
                self.nodes[i].x += 0.01
                self.nodes[i].y += 0.01

            x = [node.x for node in self.nodes]
            y = [node.y for node in self.nodes]
            scat.set_offsets(list(zip(x, y)))
            return scat,

        ani = animation.FuncAnimation(fig, update, frames=100, blit=True, repeat=True)
        writer = PillowWriter(fps=30)
        ani.save("animasyon.gif", writer=writer)
        plt.close()


class HareketModeli:
    def __init__(self, transmission_range):
        self.transmission_range = transmission_range

    def get_displacement(self, x, y):
        raise NotImplementedError()


class RastgeleHareket(HareketModeli):
    def __init__(self, transmission_range, max_step_size):
        super().__init__(transmission_range)
        self.max_step_size = max_step_size

    def get_displacement(self, x, y):
        dx = random.uniform(-self.max_step_size, self.max_step_size)
        dy = random.uniform(-self.max_step_size, self.max_step_size)
        return dx, dy

class GaussMarkov(HareketModeli):
    def __init__(self, transmission_range, alpha, beta):
        super().__init__(transmission_range)
        self.alpha = alpha
        self.beta = beta
        self.previous_dx = 0
        self.previous_dy = 0
        
    def get_displacement(self, x, y):
        dx = self.alpha*self.previous_dx + random.gauss(0, self.beta)
        dy = self.alpha*self.previous_dy + random.gauss(0, self.beta)
        self.previous_dx = dx
        self.previous_dy = dy
        return dx, dy
    
class Manzara(HareketModeli):
    def __init__(self, transmission_range, num_landmarks, landmark_range):
        super().__init__(transmission_range)
        self.landmarks = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_landmarks)]
        self.landmark_range = landmark_range
        
    def get_displacement(self, x, y):
        closest_landmark = min(self.landmarks, key=lambda l: ((l[0]-x)**2 + (l[1]-y)**2)**0.5)
        dx, dy = closest_landmark[0]-x, closest_landmark[1]-y
        dist_to_landmark = (dx**2 + dy**2)**0.5
        if dist_to_landmark <= self.landmark_range:
            return 0, 0
        dx, dy = dx*self.transmission_range/dist_to_landmark, dy*self.transmission_range/dist_to_landmark
        return dx, dy
    
class Ruzgar(HareketModeli):
    def __init__(self, transmission_range, wind_speed, wind_dir, max_step_size):
        super().__init__(transmission_range)
        self.wind_speed = wind_speed
        self.wind_dir = wind_dir
        self.max_step_size = max_step_size
        
    def get_displacement(self, x, y):
        dx_wind = self.wind_speed * math.cos(math.radians(self.wind_dir))
        dy_wind = self.wind_speed * math.sin(math.radians(self.wind_dir))
        dx_walk = random.uniform(-self.max_step_size, self.max_step_size)
        dy_walk = random.uniform(-self.max_step_size, self.max_step_size)
        dx, dy = dx_wind + dx_walk, dy_wind + dy_walk
        return dx, dy

class IzleVeHareketEt(HareketModeli):
    def __init__(self, transmission_range, max_speed):
        super().__init__(transmission_range)
        self.max_speed = max_speed
        self.current_angle = random.uniform(0, 2*math.pi)
    
    def get_displacement(self, x, y):
        speed = random.uniform(0, self.max_speed)
        dx = speed * math.cos(self.current_angle)
        dy = speed * math.sin(self.current_angle)
        self.current_angle += random.uniform(-math.pi/4, math.pi/4)
        return dx, dy

class Sismik(HareketModeli):
    def __init__(self, transmission_range, magnitude):
        super().__init__(transmission_range)
        self.magnitude = magnitude

    def get_displacement(self, x, y):
        dx = random.uniform(-self.magnitude, self.magnitude)
        dy = random.uniform(-self.magnitude, self.magnitude)
        return dx, dy