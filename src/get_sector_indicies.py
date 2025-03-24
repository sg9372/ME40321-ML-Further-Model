

def get_sector_indicies(all_sectors, sectors):
    indicies = [[] for _ in range(len(all_sectors))]
    
    for i in range(len(all_sectors)):
        for j in range(len(sectors)):
            if all_sectors[i]==sectors[j]:
                indicies[i].append(j)
    
    return indicies
