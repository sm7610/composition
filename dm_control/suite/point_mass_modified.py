# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Point-mass domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('point_mass.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the easy point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = PointMass(randomize_gains=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the hard point_mass task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = PointMass(randomize_gains=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def go_horizontal(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the easy point_mass task, with target x = 0."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = PointMass(randomize_gains=False, random=random, move_dir='horizontal')
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add()
def go_vertical(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the easy point_mass task, with target y = 0."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = PointMass(randomize_gains=False, random=random, move_dir='vertical')
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """physics for the point_mass domain."""

  def mass_to_target(self):
    """Returns the vector from mass to target in global coordinate."""
    return (self.named.data.geom_xpos['target'] -
            self.named.data.geom_xpos['pointmass'])

  def mass_to_target_x(self):
    """Returns the distance from mass to x = 0 target."""
    return np.linalg.norm((self.named.data.geom_xpos['target'][0] -
            self.named.data.geom_xpos['pointmass'][0]))

  def mass_to_target_y(self):
    """Returns the distance from mass to y = 0 target."""
    return np.linalg.norm((self.named.data.geom_xpos['target'][1] -
            self.named.data.geom_xpos['pointmass'][1]))

  def mass_to_target_dist(self):
    """Returns the distance from mass to the target."""
    return np.linalg.norm(self.mass_to_target())


class PointMass(base.Task):
  """A point_mass `Task` to reach target with smooth reward."""

  def __init__(self, randomize_gains, random=None, move_dir=None):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
      move_direction: Optional, if the target is x = 0 or y = 0. Default
        target is (x,y) = (0,0).
    """
    self._randomize_gains = randomize_gains
    super().__init__(random = random)
    self._move_dir = move_dir

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

       If _randomize_gains is True, the relationship between the controls and
       the joints is randomized, so that each control actuates a random linear
       combination of joints.

    Args:
      physics: An instance of `mujoco.Physics`.
    """
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    if self._randomize_gains:
      dir1 = self.random.randn(2)
      dir1 /= np.linalg.norm(dir1)
      # Find another actuation direction that is not 'too parallel' to dir1.
      parallel = True
      while parallel:
        dir2 = self.random.randn(2)
        dir2 /= np.linalg.norm(dir2)
        parallel = abs(np.dot(dir1, dir2)) > 0.9
      physics.model.wrap_prm[[0, 1]] = dir1
      physics.model.wrap_prm[[2, 3]] = dir2
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics, move_dir):
    """Returns a reward to the agent."""
    target_size = physics.named.model.geom_size['target', 0]
    near_target = rewards.tolerance(physics.mass_to_target_dist(),
                                    bounds=(0, target_size), margin=target_size)
    if self._move_dir == 'horizontal':
      near_target = rewards.tolerance(physics.mass_to_target_x(),
        bounds=(0, target_size), margin=target_size)
    elif self._move_dir == 'vertical':
      near_target = rewards.tolerance(physics.mass_to_target_y(), 
        bounds=(0, target_size), margin=target_size)

    control_reward = rewards.tolerance(physics.control(), margin=1,
                                       value_at_margin=0,
                                       sigmoid='quadratic').mean()
    small_control = (control_reward + 4) / 5
    return near_target * small_control
