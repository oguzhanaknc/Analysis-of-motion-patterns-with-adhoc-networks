from simulation import AdHocNetwork, RastgeleHareket, GaussMarkov, Manzara, Sismik, Ruzgar, IzleVeHareketEt

# Ad-Hoc ağ oluşturma
num_nodes = 20
adhoc_network = AdHocNetwork(num_nodes, "Sismik")

# Simülasyonu çalıştırma
num_iterations = 50
adhoc_network.run_simulation(num_iterations)

# Sonuçları yazdırma
print("Delivered messages: ", adhoc_network.delivered_msgs)
print("Total messages: ", adhoc_network.total_msgs)
print("Delivery ratio: ", adhoc_network.delivered_msgs / adhoc_network.total_msgs)