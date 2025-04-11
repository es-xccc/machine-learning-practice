import matplotlib.pyplot as plt

def getTemperatureData(fileName = "Temperature.txt"): 
    with open(fileName, 'r') as dataFile:
        city = dataFile.readline().strip()
        cityTotalYearTemperature = [city]
        for year in range(9):
            line = dataFile.readline().strip()
            yearTemperature = list(map(float, line.split(',')))
            cityTotalYearTemperature.append(yearTemperature)
    return cityTotalYearTemperature

data = getTemperatureData()

# Plot monthly mean temperature from 2013 to 2021
plt.figure()
x = list(range(1, 13))
for year_data in data[1:]:
    label = int(year_data[0])
    y_values = year_data[1:]
    plt.plot(x, y_values, label=str(label))
plt.xticks(range(1, 13))
plt.legend(loc='lower center')
plt.title("Tainan Monthly Mean Temperature From 2013 To 2021")
plt.xlabel('Month')
plt.ylabel('Temperature in Degree C')
plt.savefig("HW02_1.png")

# Plot monthly average temperature of 2013 to 2021
plt.figure()
mon_avg_temp = [round(sum(year_data[i+1] for year_data in data[1:]) / 9, 2) for i in range(12)]
allavg = round(sum(mon_avg_temp) / 12, 2)
plt.axhline(y=allavg, color='r', linestyle='--', label="Mean of 9 Years")
plt.text(1, allavg, str(allavg), fontsize=8)
plt.plot(x, mon_avg_temp, 'ro')
plt.plot(x, mon_avg_temp)
for i in range(len(x)):
    plt.text(x[i], mon_avg_temp[i], str(mon_avg_temp[i]), fontsize=8)
plt.xticks(range(1, 13))
plt.ylim(16, 32)
plt.legend(loc='upper right')
plt.title("Tainan Monthly Mean Temperature Of 2013 To 2021")
plt.xlabel('Month')
plt.ylabel('Temperature in Degree C')
plt.savefig("HW02_2.png")

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
for year_data in data[1:]:
    label = int(year_data[0])
    y_values = year_data[1:]
    plt.plot(x, y_values, label=str(label))
plt.xticks(range(1, 13))
plt.ylim(16, 32)
plt.legend(loc='lower center')
plt.title("Tainan Monthly Mean Temperature From 2013 To 2021")
plt.xlabel('Month')
plt.ylabel('Temperature in Degree C')
plt.subplot(1, 2, 2)
mon_avg_temp = [round(sum(year_data[i+1] for year_data in data[1:]) / 9, 2) for i in range(12)]
allavg = round(sum(mon_avg_temp) / 12, 2)
plt.axhline(y=allavg, color='r', linestyle='--', label="Mean of 9 Years")
plt.text(1, allavg, str(allavg), fontsize=8)
plt.plot(x, mon_avg_temp, 'ro')
plt.plot(x, mon_avg_temp)
for i in range(len(x)):
    plt.text(x[i], mon_avg_temp[i], str(mon_avg_temp[i]), fontsize=8)
plt.xticks(range(1, 13))
plt.ylim(16, 32)
plt.legend(loc='upper right')
plt.title("Tainan Monthly Mean Temperature Of 2013 To 2021")
plt.xlabel('Month')
plt.ylabel('Temperature in Degree C')
plt.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.05, wspace=0.1)
plt.savefig("HW02_3.png")