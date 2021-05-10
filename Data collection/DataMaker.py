import sc2reader
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sc2reader.engine.plugins import APMTracker
import gc

sc2reader.engine.register_plugin(APMTracker())

def commandString(self):
    string = str()
    if self.has_ability:
        if self.ability:
            string += "Ability {0}".format(self.ability.name)
    else:
        string += "Right Click"

    if self.ability_type == "TargetUnit":
        string += " - Target: {0}".format(self.target.name)

    return string


def spending_quotient(event, playerlist, playerdict):
    if event.second % 60 != 0:
        col_rate = (
            event.minerals_collection_rate + 2 * event.vespene_collection_rate
        ) / 3
        unspend_resources = (event.minerals_current + 2 * event.vespene_current) / 3
        if unspend_resources < 1:
            unspend_resources = 1
        SQ = 35 * (0.00137 * col_rate - np.log(unspend_resources)) + 240
        playerlist.append(SQ)
    else:
        col_rate = (
            event.minerals_collection_rate + 2 * event.vespene_collection_rate
        ) / 3
        unspend_resources = (event.minerals_current + 2 * event.vespene_current) / 3
        if unspend_resources < 1:
            unspend_resources = 1
        SQ = 35 * (0.00137 * col_rate - np.log(unspend_resources)) + 240
        playerlist.append(SQ)
        playerdict[f"{int(event.second/60)}"] = np.mean(playerlist)
        playerlist.clear()

def getDicID(event):
    if event not in unique:
        global dicnumber
        unique[event] = dicnumber
        dicnumber += 1
    return unique[event]


def create_league_count():
    global league_count
    league_count = {2:0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}


def check_league(league, until):
    if league_count[league] == until / 6:
        return True
    else:
        return False

def update_league(league):
    league_count[league] += 1


def create_log(until):
    if until % 6 != 0:
        newuntil = until + (6 - until % 6)
    else:
        newuntil = until
    global unique
    unique = {}
    global dicnumber
    dicnumber = 1
    ## Change with scraped replays here
    replays = sc2reader.load_replays(
        r"Data\Replays", load_level=4)
    print("Woop woop, replays read")
    global winnerlog
    winnerlog = []
    global sequencelogplayer1
    sequencelogplayer1 = []
    global sequencelogplayer2
    sequencelogplayer2 = []
    global y
    y = []
    global yrank
    yrank = []
    global rflog
    rflog = []  # final standing of playerupdate events for random forest
    matchID = 0
    global lengthlist
    lengthlist = []
    global lengthlist1
    lengthlist1 = []
    global lengthlist2
    lengthlist2 = []
    global apm1
    apm1 = []
    global apm2
    apm2 = []
    global sq1
    sq1 = []
    global sq2
    sq2 = []
    global time1
    time1 = []
    global time2
    time2 = []
    try:
        for replay in replays:
            try:
                if replay.game_length.seconds < 120: continue
                if replay.player[1].result == "Loss":
                    win = 0
                else:
                    win = 1
                    # If player two won, player 1 did not win.

                replaylog = []
                # i = 0
                # k = 1
                nextPlayer = 1
                features = []
                playerType = [0, 0]  # 1 = human , 2 = cpu, 3 = neutral, 4 = hostile
                try:
                    playerRanking = [
                        replay.player[1].highest_league,
                        replay.player[2].highest_league,
                    ]
                    if playerRanking[0] not in [2,3,4,5,6,7] and check_league(playerRanking[0], newuntil): continue
                except Exception as e:
                    continue
                updateTypes = ["Set", "Add", "Get"]
                sequences = {"sequence1": [], "sequence2": []}
                times = {"time1": [], "time2": []}

                # Needed for spending quotient.
                SQ_dict1 = {}
                SQ_dict2 = {}
                SQ_avg_list1 = []
                SQ_avg_list2 = []
                for event in replay.events:
                    if "PlayerStatsEvent" in event.name:
                        if event.pid == nextPlayer:
                            if event.pid == 1:
                                features = []
                            for key, value in event.stats.items():
                                features.append(value)
                                spending_quotient(event, SQ_avg_list1, SQ_dict1)
                            if event.pid == 2:
                                replaylog.append(features)
                                spending_quotient(event, SQ_avg_list2, SQ_dict2)
                            nextPlayer = 2 if nextPlayer == 1 else 1
                        else:
                            break
                    else:
                        if event.second < 120: continue
                        else:    
                            if "PlayerSetupEvent" in str(event.__class__):
                                playerType[event.pid - 1] = event.type

                            elif "UnitBornEvent" in str(event.__class__):
                                try:
                                    sequences[
                                        "sequence" + str(event.unit_controller.pid)
                                    ].append(getDicID(str(event.unit_type_name) + " born"))
                                    times["time" + str(event.unit_controller.pid)].append(
                                        event.second
                                    )
                                except Exception as e:
                                    if str(event.unit_controller) not in "None":
                                        print(
                                            str(e)
                                            + "unitborn "
                                            + str(event.unit)
                                            + " "
                                            + str(event.unit_controller)
                                        )

                            elif "UnitDiedEvent" in str(event.__class__):
                                try:
                                    sequences["sequence" + str(event.unit.owner.pid)].append(
                                        getDicID(str(event.unit.name) + " died")
                                    )
                                    times["time" + str(event.unit.owner.pid)].append(
                                        event.second
                                    )

                                except Exception as e:
                                    if str(event.unit.owner) not in "None":
                                        print(
                                            str(e)
                                            + "unitdied "
                                            + str(event.unit)
                                            + " "
                                            + str(event.unit.owner)
                                        )
                            elif "UnitTypeChangeEvent" in str(event.__class__):
                                try:
                                    sequences["sequence" + str(event.unit.owner.pid)].append(
                                        getDicID(
                                            str(event.unit.name)
                                            + " changed to "
                                            + str(event.unit_type_name)
                                        )
                                    )
                                    times["time" + str(event.unit.owner.pid)].append(
                                        event.second
                                    )

                                except Exception as e:
                                    pass
                            elif "UpgradeCompleteEvent" in str(event.__class__):
                                try:
                                    sequences["sequence" + str(event.player.pid)].append(
                                        getDicID(str(event.upgrade_type_name) + " upgraded")
                                    )
                                    times["time" + str(event.player.pid)].append(event.second)

                                except Exception as e:
                                    pass
                            elif "UnitInitEvent" in str(event.__class__):
                                try:
                                    sequences["sequence" + str(event.control_pid)].append(
                                        getDicID(str(event.unit_type_name) + " initiated")
                                    )
                                    times["time" + str(event.control_pid)].append(event.second)

                                except Exception as e:
                                    print(str(e) + "unitinitevent")
                            elif "UnitDoneEvent" in str(event.__class__):
                                try:
                                    sequences["sequence" + str(event.unit.owner.pid)].append(
                                        getDicID(str(event.unit.name) + " finished")
                                    )
                                    times["time" + str(event.unit.owner.pid)].append(
                                        event.second
                                    )

                                except Exception as e:
                                    print(str(e) + "unitdone")
                            elif "CommandEvent" in str(event.__class__):
                                try:
                                    sequences["sequence" + str(event.player.pid)].append(
                                        getDicID(commandString(event))
                                    )
                                    times["time" + str(event.player.pid)].append(event.second)

                                except Exception as e:
                                    print(str(e) + " commandevent" + str(event.player.pid))
                            elif "ControlGroup" in str(event.__class__):
                                try:
                                    sequences["sequence" + str(event.player.pid)].append(
                                        getDicID(
                                            updateTypes[event.update_type]
                                            + " control group "
                                            + str(event.control_group)
                                        )
                                    )
                                    times["time" + str(event.player.pid)].append(event.second)
                                except:
                                    pass

                            elif "SelectionEvent" in str(event.__class__):
                                try:
                                    sequences["sequence" + str(event.player.pid)].append(
                                        getDicID("Selection")
                                    )
                                    times["time" + str(event.player.pid)].append(event.second)
                                except Exception as e:
                                    pass

                # PLAYER1
                if len(replaylog) == 0: continue
                if len(sequences["sequence1"]) > 8472: continue
                if len(sequences[ "sequence2"]) > 8362: continue
                winnerlog.append(replaylog)
                rflog.append(replaylog[-1])
                sequencelogplayer1.append(sequences["sequence1"])
                sequencelogplayer2.append(sequences["sequence2"])
                y.append(win)
                yrank.append(playerRanking[0])
                update_league(playerRanking[0])
                lengthlist.append(len(replaylog))
                lengthlist1.append(len(sequences["sequence1"]))
                lengthlist2.append(len(sequences["sequence2"]))
                apm1.append(replay.player[1].apm)
                apm2.append(replay.player[2].apm)
                sq1.append(SQ_dict1)
                sq2.append(SQ_dict2)
                time1.append(times["time1"])
                time2.append(times["time2"])
                
                matchID += 1
                # print(f"{matchID} finished")
                if matchID % 50 == 0:
                    print(f"{matchID} reached with league count as {league_count}")
                    gc.collect()
                if matchID >= newuntil:
                    break

            except Exception as e:
                print(str(e) + str(replay.game_length) + str(replaylog) + str(len(sequences["sequence1"])))
                
    except Exception as e:
        print(str(e))


def convert_log():
    # Features
    winnerpd = pd.DataFrame(winnerlog)
    winnerpd = winnerpd.apply(
        lambda s: s.fillna({i: [0] * 78 for i in winnerpd.index})
    )
    test = winnerpd.values.tolist()
    global X
    X = np.array(test)
    del winnerpd
    del test

    global Xrf
    Xrf = np.array(rflog)

    # Sequences
    sequencepd1 = pd.DataFrame(sequencelogplayer1)
    sequencepd1 = sequencepd1.fillna(0)
    p1list = sequencepd1.values.tolist()
    global Xp1
    Xp1 = np.array(p1list).astype(int)
    del sequencepd1
    del p1list

    sequencepd2 = pd.DataFrame(sequencelogplayer2)
    sequencepd2 = sequencepd2.fillna(0)
    p2list = sequencepd2.values.tolist()
    global Xp2
    Xp2 = np.array(p2list).astype(int)
    del sequencepd2
    del p2list

    # Paper features
    apmpd1 = pd.DataFrame(apm1)
    apmpd1 = apmpd1.fillna(0)
    apm1list = apmpd1.values.tolist()
    apm1_temp = np.array(apm1list)
    del apm1list
    del apmpd1

    apmpd2 = pd.DataFrame(apm2)
    apmpd2 = apmpd2.fillna(0)
    apm2list = apmpd2.values.tolist()
    apm2_temp = np.array(apm2list)
    del apm2list
    del apmpd2

    sqpd1 = pd.DataFrame(sq1)
    sqpd1 = sqpd1.fillna(0)
    sq1list = sqpd1.values.tolist()
    sq1_temp = np.array(sq1list)
    del sq1list
    del sqpd1

    global sq2_temp
    sqpd2 = pd.DataFrame(sq2)
    sqpd2 = sqpd2.fillna(0)
    sq2list = sqpd2.values.tolist()
    sq2_temp = np.array(sq2list)
    del sq2list
    del sqpd2

    # PF = paper features until better name
    global PF1
    global PF2
    PF1 = []
    PF2 = []
    for i in range(0, len(apm1_temp)):
        PF1.append([apm1_temp[i], sq1_temp[i]])
        PF2.append([apm2_temp[i], sq2_temp[i]])
    del apm1_temp
    del apm2_temp
    del sq1_temp
    del sq2_temp
    PF1 = np.asarray(PF1)
    PF1 = np.swapaxes(PF1, 1, 2)
    PF2 = np.asarray(PF2)
    PF2 = np.swapaxes(PF2, 1, 2)

    global yrank
    yrank = np.array(yrank)

    global ywin
    ywin = np.array(y)


def export_now():
    import pickle
    global X, Xp1, Xp2, yrank, ywin, PF1, PF2, unique, time1, time2, rflog
    np.save(filepath + "\X.npy", X)
    del X
    gc.collect()
    np.save(filepath + "\Xp1.npy", Xp1)
    del Xp1
    gc.collect()
    np.save(filepath + "\Xp2.npy", Xp2)
    del Xp2
    gc.collect()    
    np.save(filepath + "\yrank.npy", yrank)
    del yrank
    gc.collect()
    np.save(filepath + "\ywin.npy", ywin)
    del ywin
    gc.collect()
    np.save(filepath + "\PF1.npy", PF1)
    del PF1
    gc.collect()
    np.save(filepath + "\PF2.npy", PF2)
    del PF2
    gc.collect()
    with open(filepath + "\unique.pkl", "wb") as handle:
        pickle.dump(unique, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del unique
    gc.collect()
    with open(filepath + "\time1.pkl", "wb") as handle:
        pickle.dump(time1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del time1
    gc.collect()
    with open(filepath + "\time2.pkl", "wb") as handle:
        pickle.dump(time2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del time2
    gc.collect()
    with open(filepath + "", "wb") as handle:
        pickle.dump(rflog, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del rflog
    gc.collect()

def main(until):#, datastring, datamap):
    filepath = input("Give the filepath of the folder where the Numpy arrays should be stored: "
    print("Start")
    create_league_count()
    print("league count created")
    create_log(until)
    print("log created")
    convert_log()
    print("log converted")
    export_now()
    print("Finished")

main(12000)
print(league_count)
