import numpy as np
import hashlib

class ABSplitter:
    def __init__(self, count_slots, salt_one, salt_two):
        self.count_slots = count_slots
        self.salt_one = salt_one
        self.salt_two = salt_two

        self.slots = np.arange(count_slots)
        self.experiments = []
        self.experiment_to_slots = dict()
        self.slot_to_experiments = dict()

    def split_experiments(self, experiments):
        """Устанавливает множество экспериментов, распределяет их по слотам.

        Нужно определить атрибуты класса:
            self.experiments - список словарей с экспериментами
            self.experiment_to_slots - словарь, {эксперимент: слоты}
            self.slot_to_experiments - словарь, {слот: эксперименты}
        experiments - список словарей, описывающих пилот. Словари содержит три ключа:
            experiment_id - идентификатор пилота,
            count_slots - необходимое кол-во слотов,
            conflict_experiments - list, идентификаторы несовместных экспериментов.
            Пример: {'experiment_id': 'exp_16', 'count_slots': 3, 'conflict_experiments': ['exp_13']}
        return: List[dict], список экспериментов, которые не удалось разместить по слотам.
            Возвращает пустой список, если всем экспериментам хватило слотов.
        """
        self.experiments = experiments
        experiments = sorted(experiments, key=lambda x: len(x['conflict_experiments']), reverse=True)

        self.slot_to_experiments = {slot: [] for slot in self.slots}
        self.experiment_to_slots = {experiment['experiment_id']: [] for experiment in experiments}
        unassigned_experiments = []
        for experiment in experiments:
            if experiment['count_slots'] > len(self.slots):
                print(f'ERROR: experiment_id={experiment["experiment_id"]} needs too many slots.')
                unassigned_experiments.append(experiment)
                continue

            # найдём доступные слоты
            notavailable_slots = []
            for conflict_experiment_id in experiment['conflict_experiments']:
                notavailable_slots += self.experiment_to_slots[conflict_experiment_id]
            available_slots = list(set(self.slots) - set(notavailable_slots))

            if experiment['count_slots'] > len(available_slots):
                print(f'ERROR: experiment_id="{experiment["experiment_id"]}" not enough available slots.')
                unassigned_experiments.append(experiment)
                continue

            # shuffle - чтобы внести случайность, иначе они все упорядочены будут по номеру slot
            # np.random.shuffle(available_slots)
            #         print(f'available_slots: {available_slots}')
            #         print(f'self.slot_to_experiments: {self.slot_to_experiments}')
            available_slots_orderby_count_experiment = sorted(
                available_slots,
                key=lambda x: len(self.slot_to_experiments[x]), reverse=True
            )
            #         print(f'available_slots_orderby_count_experiment: {available_slots_orderby_count_experiment}')

            experiment_slots = available_slots_orderby_count_experiment[:experiment['count_slots']]
            self.experiment_to_slots[experiment['experiment_id']] = experiment_slots
            for slot in experiment_slots:
                self.slot_to_experiments[slot].append(experiment['experiment_id'])
        return unassigned_experiments

    
    def process_user(self, user_id: str):
        """Определяет в какие эксперименты попадает пользователь.

        Сначала нужно определить слот пользователя.
        Затем для каждого эксперимента в этом слоте выбрать пилотную или контрольную группу.

        user_id - идентификатор пользователя.

        return - (int, List[tuple]), слот и список пар (experiment_id, pilot/control group).
            Example: (2, [('exp 3', 'pilot'), ('exp 5', 'control')]).
        """
        # YOUR_CODE_HERE
        hash_value = int(hashlib.md5(str.encode(user_id + str(self.salt_one))).hexdigest(), 16)
        slot = hash_value % self.count_slots
        assignments = []
        for experiment_id in self.slot_to_experiments[slot]:
            for experiment in self.experiments:
                if experiment_id == experiment['experiment_id']:
                    combined_id = str(user_id) + str(experiment_id) + str(self.salt_two)
                    hash_value = int(hashlib.md5(str.encode(combined_id)).hexdigest(), 16)
                    is_pilot = hash_value % 2
                    assignments.append((experiment_id, 'pilot' if is_pilot else 'control'))
        return (slot, assignments)